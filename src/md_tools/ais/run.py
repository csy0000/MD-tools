"""Annealed importance sampling: non-equilibrium switching paths from an equilibrium ensemble.

Driven by the `AIS.py` that `md-openmm build-md --config .../AIS.config` generates. The physics is
the implementation already validated on this branch -- `TauSwitcher`, and the same REST2
decomposition a fixed-tau walker or a ladder rung uses -- reached from the installed package
instead of from a copied-in script.

WHAT A PATH IS

The Hamiltonian is annealed from `tau_start` to `tau_end` while the coordinates propagate. The
temperature never changes: this is Hamiltonian switching, not temperature annealing. tau is the one
public, persisted protocol coordinate; the solute-solute and solute-environment scale factors are
derived from it inside the scaler and are deliberately not written as a second coordinate a reader
could take as authoritative.

THE WORK CONVENTION, STATED ONCE

    delta_W_j = U(tau_{j+1}, x_j) - U(tau_j, x_j)

The parameters move FIRST, at frozen coordinates; the configuration then propagates under the new
Hamiltonian. Observation 0 is the source configuration under the source Hamiltonian, before any
parameter change and before any propagation, so its cumulative work is exactly zero by definition
rather than nearly zero.

FIXED VOLUME. No barostat is active during switching and no pressure-volume term enters the work,
even when the source ensemble was NPT. Each path keeps the box of the frame it started from.

Paths are independent: each has its own directory, its own deterministic seeds derived from the run
seed and the path index, and its own completion record.

A completed path is skipped -- after its completion manifest and the sha256 of every output it
claims have been verified, because "status: completed" is a field in a file and the files it
describes are what a reader will actually load.

An interrupted path RESUMES MID-PATH, from its last committed checkpoint generation. This text
used to say the opposite -- that a switching path has no meaningful mid-path restart because the
work integral is only defined along a whole path. The premise is right and the conclusion was
wrong: the integral is defined along the whole path, and a resume continues the SAME path, with
the accumulated work, the component accumulators, the Context and every stream counter restored
to one committed instant. `md_tools.openmm.checkpoint` is what makes that instant well defined,
and `test_ais_recovery_integration.py` proves a resumed path reproduces an uninterrupted one's
totals at every transaction and stream boundary.

DECOMPOSITION. Every observation also carries the potential and the work split into the three
tau-basis groups -- see `md_tools.ais.decomposition` for the identity, the columns, the tolerances
and what the three extra energy evaluations per update cost.
"""

from __future__ import annotations

import argparse
import csv
import re
import json
import sys
from pathlib import Path
from typing import Any

from ..build.record import LogWriter, file_facts, openmm_platform_facts, read_record
from ..openmm.trajectory import check_trajectory_declaration
from .decomposition import (COMPONENT_SUMMARY_COLUMNS, DECOMPOSITION_SCHEMA, GROUPS,
                            HS_COLUMNS, OBSERVATION_POTENTIAL_COLUMNS,
                            RECONSTRUCTION_TOLERANCES, WORK_COMPONENT_COLUMNS, ComponentProbe,
                            ComponentWork, EvaluationCounters, reconstruction_tolerance,
                            require_compatible_schema)

#: One row per observation. `s` and sqrt(s) are absent on purpose: see the module docstring.
#:
#: WHICH INSTANT A ROW DESCRIBES -- and it is two different instants, which is the correction this
#: schema exists to make.
#:
#:   the WORK columns are the work of the switches since the previous emitted observation. Work is
#:   defined at the frozen pre-switch coordinate, and that is where its components are measured.
#:
#:   the OBSERVATION POTENTIAL columns are measured at the coordinate this row SAVED, under this
#:   row's tau. They are present only when `coordinate_frame_index` is, and are empty otherwise --
#:   never a neighbouring frame's values wearing this row's frame index.
#:
#: The previous schema put the pre-switch work basis in columns that read as potentials at the
#: saved frame, one propagation earlier than the coordinate they named. A Hummer-Szabo
#: reweighting from those rows pairs the work of one configuration with the energy of another.
OBSERVATION_COLUMNS = (
    "path_index", "observation_index", "coordinate_frame_index",
    "source_frame_index", "protocol_step", "switching_time_ps", "tau",
    "incremental_work_kj_mol", "cumulative_work_kj_mol", "cumulative_reduced_work",
    "temperature_kelvin", "integrator_seed", "velocity_seed",
) + WORK_COMPONENT_COLUMNS + OBSERVATION_POTENTIAL_COLUMNS

COMPLETION_NAME = "completed.json"
OBSERVATIONS_CSV = "observations.csv"

#: The per-path thermodynamic state table, at `reporting.system_printout`. A different question
#: from the work: how the path is BEHAVING while the Hamiltonian moves, which is what tells you a
#: switch is too fast long before the work distribution does.
STATE_CSV = "system.csv"
STATE_COLUMNS = ("path_index", "protocol_step", "switching_time_ps", "tau",
                 "potential_energy_kj_mol", "kinetic_energy_kj_mol", "total_energy_kj_mol",
                 "temperature_kelvin", "volume_nm3", "density_g_per_ml")

#: Mid-path resume lives in `md_tools.openmm.checkpoint`: a generation-based transaction whose
#: committed pointer is replaced last, so a crash never pairs a new Context with old bookkeeping.
#: `RESUME_SIDECAR` and `PATH_CHECKPOINT` were the two files of the previous, non-atomic design.
#: Frames are staged inside the path directory and published to the run root only when the path is
#: complete and validated. A half-written `AIS_trajNNNN.nc` at the root would look exactly like a
#: finished path to anyone globbing the directory.
STAGED_TRAJECTORY = "frames.partial.nc"

#: The run-level identity of an AIS output directory, written once and never rewritten.
#:
#: A path's checkpoint fingerprint protects that path. Nothing protected the DIRECTORY: a second
#: invocation into the same `-odir` with a different source trajectory, a different tau schedule,
#: a different seed or a different path count would skip the completed paths, run the rest under
#: the new settings, and assemble one work table out of two different measurements. The table
#: reads perfectly and describes no experiment.
RUN_IDENTITY = "AIS_run.json"

#: The schema of that record. Bumped when the SET of fields changes, so a directory written by an
#: older build is refused by name rather than compared field by field against a shape it never had.
RUN_IDENTITY_VERSION = 1

#: The global work table rank 0 writes once every path this run owns has finished.
#:
#: One row per (path, switching step), keyed by exactly that pair and sorted by it, so the table
#: is the same file whatever the worker count was and whichever rank produced which row -- see
#: `paths_for_rank`. It is assembled from the per-path `observations.csv` files rather than
#: gathered over MPI: a rank that died leaves its finished paths on disk, so the table describes
#: exactly what was measured, and a resumed run rebuilds it from the same records.
WORK_TABLE = "AIS_work.csv"
#:
#: `delta_work_kj_mol` is the work of every switch since the previous emitted work observation --
#: one switch when the work and switching cadences agree, and their ratio otherwise. Stated here
#: and in `DECOMPOSITION_SCHEMA["delta_work_meaning"]` rather than left for a reader to infer from
#: two interval settings.
WORK_COLUMNS = ("path_id", "source_frame", "observation_index", "switch_step",
                "coordinate_frame_index", "tau_before", "tau_after",
                "delta_work_kj_mol", "total_work_kj_mol", "total_reduced_work",
                "trajectory", "mpi_rank") + WORK_COMPONENT_COLUMNS + OBSERVATION_POTENTIAL_COLUMNS

#: The frame-aligned Hummer-Szabo table: every row has a saved coordinate, and its potentials were
#: recomputed at that coordinate. A reader doing HS reweighting wants exactly these rows and would
#: otherwise have to filter `AIS_work.csv` themselves -- and the failure mode of forgetting to is
#: silent.
HS_TABLE = "AIS_hs.csv"

#: One row per path: the summary a reader wants when the question is about the work DISTRIBUTION
#: rather than about any individual path's trajectory through it.
WORK_SUMMARY = "AIS_paths.csv"
SUMMARY_COLUMNS = ("path_index", "source_frame_index", "observations",
                   "total_work_kj_mol", "total_reduced_work", "trajectory",
                   "integrator_seed", "velocity_seed", "mpi_rank") + COMPONENT_SUMMARY_COLUMNS


def ais_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("-p", "--topology", required=True, metavar="PDB",
                        help="topology and reference coordinates (built.pdb)")
    parser.add_argument("-s", "--system", required=True, metavar="XML",
                        help="serialised OpenMM System (built.xml)")
    parser.add_argument("-source-traj", "-src", "--source", dest="source", default=None,
                        metavar="TRAJ",
                        help="the equilibrium source trajectory these paths are drawn from; "
                             "overrides ais_source.trajectory. AIS consumes an ensemble you have "
                             "already produced -- it does not generate one")
    parser.add_argument("-odir", "--out-dir", default=".", metavar="DIR",
                        help="where the path directories are written (default: here)")
    parser.add_argument("-log", "--log", default=None, metavar="LOG",
                        help="the provenance record (machine-readable). Defaults to AIS.log")
    parser.add_argument("-o", "--output", default=None, metavar="OUT",
                        help="human-readable simulation output, Amber's mdout. A DIFFERENT file "
                             "from -log. Defaults to AIS.out")
    parser.add_argument("-ng", "--number-of-groups", dest="number_of_groups", type=int,
                        default=None, metavar="N",
                        help="how many workers share the paths. Checked against the MPI "
                             "communicator; it never changes how many paths there are")
    parser.add_argument("--resume", action="store_true",
                        help="continue interrupted paths from their checkpoints. A completed "
                             "path is skipped either way; this affects only paths that stopped "
                             "part-way")
    parser.add_argument("--overwrite", action="store_true",
                        help="start this -odir over as a NEW run: replace the recorded run "
                             "identity and every path in it. Without this, an -odir that already "
                             "holds a different run -- a different source, schedule, seed or path "
                             "count -- is refused rather than half-extended")
    # Accepted by the parser and REFUSED by the preflight, by name and with the reason. These
    # are Amber flags that other protocols implement, and a person who has just run a stage will
    # try them. Leaving them off the parser makes argparse say "unrecognized arguments", which
    # names the flag and explains nothing; worse, it means the refusal lives in argparse rather
    # than in the shared preflight, so `md-run` and `AIS.py` could disagree about it -- and they
    # did, because only `md-run` checked -x at all.
    for flag, alias, destination, why in (
            ("-x", "--trajectory", "trajectory",
             "AIS writes one trajectory per path, AIS_traj0000.nc.., named from the global path "
             "index; one -x cannot name N files"),
            ("-c", "--coordinates", "coordinates",
             "each path starts from its own frame of -source-traj, chosen by the recorded seed"),
            ("-r", "--restart", "restart",
             "each path writes its own final_state.xml inside its path directory"),
            ("-chk", "--checkpoint", "checkpoint",
             "AIS checkpoints are per-path generations under path_NNNN/checkpoints/, committed "
             "through an atomic pointer")):
        parser.add_argument(flag, alias, dest=destination, default=None, metavar="PATH",
                            help=f"refused for AIS: {why}")
    parser.add_argument("-groupfile", "--groupfile", dest="groupfile", default=None,
                        metavar="FILE",
                        help="refused for AIS: a group file is one line per replica, and AIS "
                             "paths are distributed by `paths_for_rank`, not enumerated in a file")
    parser.add_argument("--cpu", action="store_true",
                        help="run this invocation on the OpenMM CPU platform, overriding "
                             "machine.openmm.platform. That setting can also select "
                             "CPU machine-wide; this flag is the per-run override, "
                             "and the record distinguishes the two")
    parser.add_argument("--device", default=None, metavar="N",
                        help="CUDA device index. An execution placement option; rejected with "
                             "--cpu, which has no device to place")
    parser.add_argument("--paths", default=None,
                        help="run only these path indices, e.g. 0,1,2 or 0-9")
    parser.add_argument("--check", action="store_true",
                        help="validate inputs, source and schedule, then exit without switching")
    return parser


def _selected(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    wanted: list[int] = []
    for piece in str(spec).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            first, last = piece.split("-", 1)
            wanted.extend(range(int(first), int(last) + 1))
        else:
            wanted.append(int(piece))
    out = sorted({index for index in wanted if 0 <= index < total})
    if not out:
        raise SystemExit(f"--paths {spec!r} selected no path in range 0..{total - 1}")
    return out


def choose_frames(*, eligible: list[int], count: int, selection: str, allow_repeats: bool,
                  seed: int) -> list[int]:
    """Which source frame each path starts from.

    Refuses rather than quietly reusing frames: two paths from the same configuration are not two
    independent realisations, and treating them as such understates the spread of the work
    distribution the whole method exists to measure.
    """
    import random

    if not eligible:
        raise SystemExit("the source window contains no eligible frames")
    if selection == "evenly_spaced":
        if count > len(eligible) and not allow_repeats:
            raise SystemExit(
                f"{count} paths were requested from {len(eligible)} eligible frames with "
                f"evenly_spaced selection. Widen the window, ask for fewer paths, or set "
                f"ais_source.allow_repeated_frames.")
        stride = max(1, len(eligible) // max(1, count))
        chosen = [eligible[min(index * stride, len(eligible) - 1)] for index in range(count)]
        return chosen
    generator = random.Random(seed)
    if allow_repeats:
        return [generator.choice(eligible) for _ in range(count)]
    if count > len(eligible):
        raise SystemExit(
            f"{count} independent paths were requested but the source window holds only "
            f"{len(eligible)} eligible frame(s). Two paths from one configuration are not two "
            f"independent realisations. Widen ais_source.first_frame/last_frame, ask for fewer "
            f"paths, or set ais_source.allow_repeated_frames if repeats are genuinely intended.")
    return sorted(generator.sample(eligible, count))


def _source_atom_count(path: Path) -> int | None:
    """How many atoms the trajectory FILE says it holds, or None when its format does not say.

    Deliberately reads the file's own header rather than trusting the topology it will be paired
    with: the whole point is to catch the case where those two disagree.
    """
    import mdtraj

    try:
        with mdtraj.open(str(path)) as handle:
            first = handle.read(1)
    except Exception:                                  # noqa: BLE001 - format cannot say; not fatal
        return None
    xyz = first[0] if isinstance(first, tuple) else first
    try:
        return int(xyz.shape[1])
    except (AttributeError, IndexError):
        return None


def _summed_counters(completed) -> dict[str, Any]:
    """Add up every path's counters into one record for the run.

    `discarded_is_complete` is AND-ed: one path that was resumed makes the run's discarded figure
    incomplete, and a run-level summary that claimed completeness because most paths had it would
    be the least useful kind of wrong.
    """
    total = EvaluationCounters()
    for record in completed:
        total = total + EvaluationCounters.from_record(record.get("evaluation_counters"))
    return total.record()


def write_work_table(out: Path, chosen: list[int]) -> dict[str, Any]:
    """Assemble the global work table and the per-path summary from what the paths wrote.

    Read from files rather than gathered over MPI, so an interrupted campaign still produces a
    table that describes exactly the paths that finished. A path with no completion record is
    absent: the table never invents a row for work that was not measured.
    """
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    hs_rows: list[dict[str, Any]] = []
    for path_id in range(len(chosen)):
        directory = out / f"path_{path_id:04d}"
        marker = directory / COMPLETION_NAME
        if not marker.is_file():
            continue
        record = json.loads(marker.read_text(encoding="utf-8"))
        summary.append({name: record.get(name) for name in SUMMARY_COLUMNS})

        observations = directory / OBSERVATIONS_CSV
        if not observations.is_file():
            continue
        with observations.open(newline="") as handle:
            entries = list(csv.DictReader(handle))
        previous_tau = None
        for entry in entries:
            row = {
                "path_id": path_id,
                "source_frame": entry["source_frame_index"],
                "observation_index": entry["observation_index"],
                "switch_step": int(entry["protocol_step"]),
                "coordinate_frame_index": entry.get("coordinate_frame_index", ""),
                # `tau_before` on observation 0 is its own tau: the path has not moved yet, and
                # writing a blank there would make the first row the only one a reader has to
                # treat specially.
                "tau_before": previous_tau if previous_tau is not None else entry["tau"],
                "tau_after": entry["tau"],
                "delta_work_kj_mol": entry["incremental_work_kj_mol"],
                "total_work_kj_mol": entry["cumulative_work_kj_mol"],
                "total_reduced_work": entry["cumulative_reduced_work"],
                "trajectory": record.get("trajectory"),
                "mpi_rank": record.get("mpi_rank"),
            }
            # Copied by name from the per-path row rather than recomputed. The global table is an
            # assembly of what the paths measured; recomputing a column here would let the two
            # files disagree about the same number, and the one a reader trusts would be whichever
            # they opened first.
            for column in WORK_COMPONENT_COLUMNS + OBSERVATION_POTENTIAL_COLUMNS:
                row[column] = entry.get(column, "")
            rows.append(row)

            # The frame-aligned subset: only rows whose coordinate was actually saved, so every
            # HS row's potentials and work describe ONE configuration. A row without a frame has
            # empty potential cells by schema, and including it here would put those empty cells
            # in front of a reweighting that has no way to notice.
            if str(entry.get("coordinate_frame_index", "")).strip() != "":
                hs_rows.append({name: row.get(name, entry.get(name, ""))
                                for name in HS_COLUMNS})
            previous_tau = entry["tau"]

    rows.sort(key=lambda row: (row["path_id"], row["switch_step"]))
    hs_rows.sort(key=lambda row: (row["path_id"], int(row["switch_step"])))

    # ATOMIC. These tables are rewritten from scratch every time rank 0 assembles them, and
    # opening the real path with "w" truncates it first: a reader arriving during the rewrite --
    # or a crash in the middle of one -- finds a table that is valid CSV and short, which is the
    # one failure a table cannot signal. Written to a temporary and moved into place instead.
    import io

    for path, columns, payload in ((out / WORK_TABLE, WORK_COLUMNS, rows),
                                   (out / WORK_SUMMARY, SUMMARY_COLUMNS, summary),
                                   (out / HS_TABLE, HS_COLUMNS, hs_rows)):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(payload)
        write_atomically(path, buffer.getvalue())
    return {"rows": len(rows), "paths": len(summary), "requested": len(chosen),
            "hs_rows": len(hs_rows)}


def run_identity_document(*, fingerprint, topology_facts, system_facts, source_facts,
                          source_format, schedule, ais, dynamics, chosen, reporting,
                          resolved_config) -> dict[str, Any]:
    """Everything an output directory may not change between invocations.

    Deliberately NOT the same thing as the per-path fingerprint. That one answers "may this
    checkpoint be loaded into this path?"; this one answers "do these paths belong to the same
    experiment?", and the second question is the one a work table assembled from N directories
    depends on.
    """
    from .decomposition import DECOMPOSITION_SCHEMA

    return {
        "schema": "md-ais-run-identity",
        "schema_version": RUN_IDENTITY_VERSION,
        "fingerprint": fingerprint,
        "topology": {"name": Path(topology_facts["path"]).name if "path" in topology_facts
                     else None, "sha256": topology_facts["sha256"]},
        "system": {"sha256": system_facts["sha256"]},
        "source": {"sha256": source_facts["sha256"], "format": source_format},
        "tau": {"start": float(ais["tau_start"]), "end": float(ais["tau_end"]),
                "interpolation": "linear"},
        "schedule": {k: v for k, v in schedule.items()
                     if k not in ("observations", "taus", "note")},
        "reporting": {k: int(reporting[k]) for k in sorted(reporting)},
        "seed_policy": {"seed": int(dynamics["seed"]),
                        "derivation": "derive_seed(seed, 'ais', path_index, role)"},
        "number_of_paths": int(ais["number_of_paths"]),
        "selected_frames": [int(f) for f in chosen],
        "observation_columns": list(OBSERVATION_COLUMNS),
        "decomposition_schema": {"name": DECOMPOSITION_SCHEMA["name"],
                                 "version": DECOMPOSITION_SCHEMA["version"]},
        "resolved_config": resolved_config,
    }


def require_same_run(out: Path, document: dict[str, Any]) -> None:
    """Refuse an `-odir` that already belongs to a different AIS run.

    Written by rank 0 the first time and compared on every invocation after. The comparison is
    field by field so the refusal names WHAT changed -- "this directory holds a different run" is
    true and useless, and the person reading it has to diff two configurations by hand.
    """
    path = Path(out) / RUN_IDENTITY
    if not path.is_file():
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        raise SystemExit(
            f"{path} is not readable JSON. It is the record of which run this directory holds, "
            f"so without it nothing here can be matched to anything. Move the directory aside "
            f"rather than adding paths to it.") from None

    if existing.get("schema_version") != RUN_IDENTITY_VERSION:
        raise SystemExit(
            f"{path} was written under run-identity schema "
            f"v{existing.get('schema_version')}, and this build writes "
            f"v{RUN_IDENTITY_VERSION}. The set of fields differs, so 'unchanged' cannot be "
            f"established. Start a new directory rather than extending this one.")

    differing = sorted(key for key in set(existing) | set(document)
                       if existing.get(key) != document.get(key))
    differing = [key for key in differing if key != "resolved_config"]
    if not differing:
        return
    detail = "\n".join(
        f"    {key}: recorded {json.dumps(existing.get(key))[:120]}\n"
        f"    {' ' * len(key)}  now      {json.dumps(document.get(key))[:120]}"
        for key in differing[:6])
    raise SystemExit(
        f"{Path(out)} already holds a DIFFERENT AIS run, differing in "
        f"{', '.join(differing)}:\n{detail}\n"
        f"  Completed paths in this directory would be skipped and the remaining ones run under "
        f"the new settings, and the work table would then be assembled out of two different "
        f"experiments -- readable, and describing neither. Use a new -odir.")


#: The EXACT names an AIS run owns. A directory is removed only if it matches this, and a file
#: only if it matches the trajectory schema or is one of the named tables.
PATH_DIRECTORY = re.compile(r"^path_(\d{4})$")
PATH_TRAJECTORY = re.compile(r"^AIS_traj(\d{4})\.nc$")


def clear_run_directory(out: Path, *, identity: dict[str, Any] | None = None, paths: int = 0,
                        ranks: int = 1) -> int:
    """Remove every artefact THIS RUN OWNS from `out`. Returns how many were removed.

    What `--overwrite` has to mean: a fresh run must inherit nothing -- not a path directory with
    its checkpoints and manifest, not a published trajectory, not the selected-frame table, not
    an aggregate table, not a rank report, not the run identity. Any one surviving lets the new
    identity adopt an old measurement.

    OWNERSHIP IS BY NAME SCHEMA, NOT BY PREFIX. The previous version globbed `path_*` and deleted
    every directory that matched, which would take `path_notes/` -- somebody's working directory
    that happens to start with those five characters -- with it. `--overwrite` is not a licence to
    delete a directory because its name is suggestive. Only `path_0000`-style names (exactly four
    digits) and `AIS_trajNNNN.nc` are owned, plus the named tables and the rank reports this run's
    world size implies.

    Stragglers from a LONGER previous run are removed too -- a 100-path directory overwritten by a
    4-path one leaves `AIS_traj0004.nc..0099.nc`, and those look exactly like this run's own
    output -- but only because they match the schema, not because of a wildcard.
    """
    import shutil

    from ..remd.executor import report_path_for_rank

    removed = 0
    out = Path(out)
    if not out.is_dir():
        return 0

    for entry in sorted(out.iterdir()):
        if entry.is_dir() and PATH_DIRECTORY.match(entry.name):
            shutil.rmtree(entry)
            removed += 1
        elif entry.is_file() and PATH_TRAJECTORY.match(entry.name):
            entry.unlink()
            removed += 1

    for name in (RUN_IDENTITY, WORK_TABLE, WORK_SUMMARY, HS_TABLE,
                 "selected_source_frames.csv"):
        target = out / name
        if target.is_file():
            target.unlink()
            removed += 1

    # Rank reports, INCLUDING those of a previous, larger world. An overwrite that dropped from
    # six ranks to two left `AIS.out.rank05` beside the new `AIS.out`, equally current-looking and
    # describing a run that no longer exists.
    for base in ("AIS.out", "AIS.log"):
        for candidate in sorted(out.glob(f"{base}*")):
            if candidate.is_file():
                candidate.unlink()
                removed += 1
    del report_path_for_rank, ranks
    return removed


def write_atomically(path: Path, text: str) -> None:
    """Write through a temporary and `os.replace`, so a reader never sees half a file.

    The global tables are rewritten from scratch every time rank 0 assembles them. Truncating the
    real file first means any reader -- or any crash -- during the rewrite finds a table that is
    valid CSV and short, which is the one failure mode a table cannot signal.
    """
    import os

    path = Path(path)
    staging = path.with_name(path.name + ".partial")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def write_selected_frames(path: Path, chosen: list[int], *, seed: int) -> None:
    """Which source frame each path starts from, and with which seeds. Written before dynamics."""
    from ..md._stages import derive_seed

    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_index", "source_frame_index", "integrator_seed", "velocity_seed"])
        for index, frame in enumerate(chosen):
            writer.writerow([index, frame,
                             derive_seed(seed, "ais", index, "integrator"),
                             derive_seed(seed, "ais", index, "velocity")])



# ---------------------------------------------------------------------------------------------
# One switching path
#
# Four cadences, a staged trajectory, and a checkpoint that restores the OpenMM Context and the
# bookkeeping OUTSIDE it together. Restoring one without the other resumes a simulation into
# somebody else's accounting, which is worse than restarting.
# ---------------------------------------------------------------------------------------------

def _truncate_csv(path: Path, keep: int, columns) -> None:
    """Cut a table back to `keep` data rows. A crash can leave a partial final line."""
    if not path.is_file():
        with path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(columns)).writeheader()
        return
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows[:keep])


def _open_staged_netcdf(path: Path, keep: int):
    """An open NetCDF writer positioned after exactly `keep` good frames.

    mdtraj's NetCDF writer offers 'r' and 'w' and no append mode, and that turns out to be the
    right constraint rather than an obstacle: a frame interrupted mid-write leaves a record that
    reads as present and is not complete, so appending onto an unverified tail would be wrong even
    where the format allowed it.

    So a resume REWRITES: the frames the sidecar vouches for are read back, written into a fresh
    file, and the handle is returned still open for the rest of the path. The count is bounded by
    `number_of_frames`, which is small by construction, and it happens once per interruption.
    """
    import os

    import mdtraj

    if not keep:
        return mdtraj.formats.NetCDFTrajectoryFile(str(path), "w")

    with mdtraj.formats.NetCDFTrajectoryFile(str(path)) as handle:
        available = len(handle)
        coordinates, time, lengths, angles = handle.read(min(keep, available))
    if available < keep:
        raise SystemExit(
            f"{path} holds {available} frame(s) but the checkpoint sidecar vouches for {keep}. "
            f"The staged trajectory and the record of it disagree, so neither can be trusted; "
            f"delete the path directory to rerun it from its source frame.")

    staged = path.with_name(path.name + ".rewrite")
    writer = mdtraj.formats.NetCDFTrajectoryFile(str(staged), "w")
    for frame in range(keep):
        writer.write(coordinates[frame],
                     time=None if time is None else time[frame],
                     cell_lengths=None if lengths is None else lengths[frame],
                     cell_angles=None if angles is None else angles[frame])
    writer.flush()
    # `os.replace` while the handle is open is safe on POSIX: the writer keeps writing to the same
    # inode, now reachable under the staged name.
    os.replace(staged, path)
    return writer


def _state_row(simulation, *, index, protocol_step, switching_time_ps, tau, implicit,
               temperature_unit) -> dict[str, Any]:
    """What the path is doing thermodynamically at this step."""
    from openmm import unit

    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    # Degrees of freedom from the integrator's own view of the System, so constraints and any
    # centre-of-mass motion remover are accounted for rather than assumed away.
    dof = simulation.context.getIntegrator().computeSystemTemperature() \
        if hasattr(simulation.context.getIntegrator(), "computeSystemTemperature") else None
    if dof is not None:
        temperature_now = dof.value_in_unit(unit.kelvin)
    else:                                                 # older OpenMM: derive it
        particles = simulation.system.getNumParticles()
        constraints = simulation.system.getNumConstraints()
        free = max(3 * particles - constraints - 3, 1)
        gas = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
        temperature_now = 2.0 * kinetic / (free * gas)

    volume = density = None
    if not implicit:
        box = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3)
        volume = box
        mass = sum(simulation.system.getParticleMass(i).value_in_unit(unit.dalton)
                   for i in range(simulation.system.getNumParticles()))
        # g/mL from daltons per nm^3. One dalton is 1.66053906660e-24 g and one nm^3 is
        # 1e-21 mL, so the factor is their ratio: 1.66053906660e-3. Written out rather than
        # given as a bare constant, because the plausible wrong answer here is 1000x and water
        # at 947 g/mL looks like a number rather than like a mistake.
        density = mass / box * 1.66053906660e-3
    return {
        "path_index": index,
        "protocol_step": protocol_step,
        "switching_time_ps": switching_time_ps,
        "tau": tau,
        "potential_energy_kj_mol": potential,
        "kinetic_energy_kj_mol": kinetic,
        "total_energy_kj_mol": potential + kinetic,
        "temperature_kelvin": temperature_now,
        "volume_nm3": volume,
        "density_g_per_ml": density,
    }


def _verified_completion(marker: Path, *, directory: Path, published: Path, fingerprint: str,
                         index: int, frame: int, trajectory_name: str,
                         schedule: dict[str, Any]) -> dict[str, Any]:
    """Read a `completed.json` and REFUSE it unless everything it claims still holds.

    A completed path is skipped, which means this record is the only thing standing between a
    reader and a path nobody will ever look at again. Every check here is a way a directory has
    read as finished while not being:

      an old record, from before the fields below existed, which cannot be checked at all;
      a record from another run, copied or resumed into the wrong directory;
      outputs that no longer match their recorded digests -- truncated by a full disk, or
      rewritten by a second process;
      counts that do not match the schedule this invocation resolved.
    """
    from .decomposition import require_compatible_schema

    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except ValueError:
        raise SystemExit(
            f"{marker} is not readable JSON. It is the only record that path {index} finished, "
            f"so it cannot be believed and cannot be repaired. Delete "
            f"{directory} to rerun this path from its source frame.") from None

    missing = [key for key in ("fingerprint", "outputs", "source_frame_index", "observations",
                               "frames", "total_work_kj_mol") if key not in record]
    if missing:
        raise SystemExit(
            f"{marker} was written before this build's completion manifest existed: it carries "
            f"no {', '.join(missing)}. Nothing about it can be verified, so it is refused rather "
            f"than trusted. Delete {directory} to rerun this path.")
    require_compatible_schema(record, what=str(marker))

    for field, expected in (("fingerprint", fingerprint), ("path_index", index),
                            ("source_frame_index", frame), ("trajectory", trajectory_name)):
        if record.get(field) != expected:
            raise SystemExit(
                f"{marker} records {field} = {record.get(field)!r} but this run has "
                f"{expected!r}. This directory holds a path from a different run. Refusing to "
                f"skip it as though it were this one.")

    claimed = {"trajectory": published,
               "observations": directory / OBSERVATIONS_CSV,
               "final_state": directory / "final_state.xml",
               "system_table": directory / STATE_CSV}
    for name, facts in (record.get("outputs") or {}).items():
        where = claimed.get(name)
        if where is None:
            raise SystemExit(
                f"{marker} claims an output named {name!r} that this build does not produce. "
                f"Refusing to skip a path whose manifest describes a different schema.")
        if not where.is_file():
            raise SystemExit(
                f"{marker} says path {index} completed, but its {name} ({where}) is missing. "
                f"Delete {directory} to rerun this path.")
        now = file_facts(where)["sha256"]
        if now != facts.get("sha256"):
            raise SystemExit(
                f"{marker} says path {index} completed with {name} sha256 "
                f"{str(facts.get('sha256'))[:16]}..., and {where.name} now hashes to "
                f"{now[:16]}.... It has changed since the path finished, so the record describes "
                f"a file that no longer exists. Delete {directory} to rerun this path.")

    for field, expected in (("observations", schedule["number_of_observations"]),
                            ("frames", schedule["number_of_frames"])):
        if int(record[field]) != int(expected):
            raise SystemExit(
                f"{marker} records {record[field]} {field} but this run's schedule calls for "
                f"{expected}. The path was run under a different schedule.")
    return record


def run_one_path(*, index: int, chosen: list[int], out: Path, schedule: dict[str, Any],
                 taus, switcher, simulation_inputs: dict[str, Any], dynamics: dict[str, Any],
                 ais: dict[str, Any], beta: float, temperature: float, rank: int,
                 resume: bool, fingerprint: str, log) -> dict[str, Any] | None:
    """Run (or finish) one switching path. Returns its completion record, or None if it failed."""
    import os

    import mdtraj
    from openmm import LangevinMiddleIntegrator, XmlSerializer, unit
    from openmm.app import Simulation

    from ..md._stages import derive_seed
    from . import path_trajectory_name

    topology = simulation_inputs["topology"]
    source_path = simulation_inputs["source_path"]
    top = simulation_inputs["mdtraj_top"]
    implicit = simulation_inputs["implicit"]
    acceleration = simulation_inputs["acceleration"]

    directory = out / f"path_{index:04d}"
    marker = directory / COMPLETION_NAME
    trajectory_name = path_trajectory_name(index, len(chosen))
    published = out / trajectory_name

    if marker.is_file():
        # A completion record is only believed once its claims are checked. "status: completed" is
        # a field in a file; the outputs it describes are what a reader will actually load, and a
        # path whose trajectory was truncated by a full disk, or whose directory was copied from
        # another run, is exactly the case that reads as finished and is not.
        record = _verified_completion(marker, directory=directory, published=published,
                                      fingerprint=fingerprint, index=index, frame=chosen[index],
                                      trajectory_name=trajectory_name, schedule=schedule)
        log(f"  path {index:4d}: already completed and verified; not rerun and never appended to")
        return record
    directory.mkdir(parents=True, exist_ok=True)

    frame = chosen[index]
    integrator_seed = derive_seed(int(dynamics["seed"]), "ais", index, "integrator")
    velocity_seed = derive_seed(int(dynamics["seed"]), "ais", index, "velocity")

    interval = schedule["parameter_update_interval_steps"]
    observe_every = schedule["observation_interval_steps"]
    frame_every = schedule["trajectory_interval_steps"]
    state_every = schedule["state_interval_steps"]
    checkpoint_every = schedule["checkpoint_interval_steps"]
    updates = schedule["number_of_updates"]

    staged = directory / STAGED_TRAJECTORY

    system = switcher.prepared_system(taus[0])
    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        float(dynamics["friction_per_ps"]) / unit.picosecond,
        float(dynamics["timestep_fs"]) * unit.femtosecond)
    integrator.setRandomNumberSeed(int(integrator_seed))
    simulation = Simulation(topology, system, integrator,
                            acceleration.platform, acceleration.properties)

    # The component probe drives the SAME switcher the path switches with, so a probe amplitude
    # and a real tau reach this Context through one implementation.
    counters = EvaluationCounters()
    probe = ComponentProbe(switcher, system, counters=counters)
    precision = (acceleration.properties or {}).get("Precision", "mixed")

    def direct_potential() -> float:
        return simulation.context.getState(getEnergy=True).getPotentialEnergy(
            ).value_in_unit(unit.kilojoule_per_mole)

    def measure_components(tau: float, *, what: str, observation: bool = False):
        """The three basis components at the CURRENT coordinates, checked against U(tau).

        `what` names which coordinate this is -- the frozen pre-switch one, or the saved
        observation one -- so a refusal says which of the two probes disagreed. They are the same
        arithmetic at different configurations, and confusing them is precisely the defect this
        separation exists to prevent.

        The reconstruction is compared with a directly measured potential every time it is taken.
        That comparison is the whole safety net: it is what would fail if a force ever acquired a
        tau dependence outside the three-group model, and catching that at the first update is the
        difference between a refused run and a work integral for a Hamiltonian nothing ran under.
        """
        measured = direct_potential()
        if observation:
            counters.observation_potential_energy_evaluations += 1
        else:
            counters.direct_work_energy_evaluations += 1
        components = probe.measure(simulation.context, restore_tau=tau, observation=observation)
        reconstructed = components.total_at(tau)
        allowed = reconstruction_tolerance(measured, precision=precision)
        if abs(reconstructed - measured) > allowed:
            raise SystemExit(
                f"path {index}: the lambda-basis decomposition does not reproduce the potential "
                f"at tau = {tau}, measured at {what}. Direct {measured:.6f} kJ/mol, reconstructed "
                f"{reconstructed:.6f} kJ/mol from "
                f"U_non_scaled = {components.non_scaled:.6f}, "
                f"U_sqrt_scaled = {components.sqrt_scaled:.6f}, "
                f"U_lin_scaled = {components.lin_scaled:.6f}; the difference "
                f"{reconstructed - measured:.3e} exceeds the {precision}-precision tolerance "
                f"{allowed:.3e}.\n"
                f"  U(tau) is a quadratic in (1 - tau) only if every force scales as the REST2 "
                f"convention says. Refusing rather than recording components that do not add up "
                f"to the potential the path is actually running under.")
        return components, measured

    def observe_potentials(tau: float, *, wrote_frame: bool):
        """The OBSERVATION probe: the basis at the coordinate this row is about to save.

        Returns the five observation-potential columns, or empty strings when this row saved no
        coordinate. Empty rather than a neighbouring frame's numbers: a row that names no frame
        has no configuration for a potential to belong to, and filling those cells anyway is
        exactly how the previous schema came to pair one configuration's work with another's
        energy.
        """
        if not wrote_frame:
            return {name: "" for name in OBSERVATION_POTENTIAL_COLUMNS}
        components, measured = measure_components(tau, what="the saved observation coordinate",
                                                  observation=True)
        return components.row(tau, measured)

    # -- resume, or start ----------------------------------------------------------------------
    from ..openmm.checkpoint import (clear_committed, commit_generation, fault,
                                     read_committed)

    state_of_path = None
    if resume:
        # ONLY the committed pointer. Never "the newest generation on disk": the newest file is
        # exactly what a crash leaves behind, and choosing it would pair a Context from step 3000
        # with bookkeeping from step 2000.
        committed = read_committed(directory)
        state_of_path = committed["state"] if committed else None

    # `--resume` over a path that never committed a generation is not an error: that path simply
    # starts from its source frame. Only a path that HAS a committed checkpoint is fingerprinted.
    if state_of_path is not None:
        # EVERY fingerprint before anything is loaded. A checkpoint carries positions and
        # velocities for one particular System, schedule and path; resuming it against another is
        # how a run continues with right-looking numbers and the wrong simulation.
        mismatched = [key for key, expected in
                      (("fingerprint", fingerprint), ("path_index", index),
                       ("source_frame_index", frame), ("integrator_seed", integrator_seed),
                       ("velocity_seed", velocity_seed), ("trajectory", trajectory_name))
                      if state_of_path.get(key) != expected]
        if mismatched:
            raise SystemExit(
                f"path {index}: the checkpoint in {directory} does not belong to this run "
                f"({', '.join(mismatched)} differ). Refusing to resume it: the numbers would look "
                f"right and describe a different simulation. Delete the path directory to rerun "
                f"it from its source frame.")

    if state_of_path is not None:
        simulation.loadCheckpoint(committed["checkpoint"])
        # The Context is back; now put the bookkeeping back to exactly the same instant. The
        # streams are cut to the counts the sidecar vouches for, so an interrupted final record
        # is dropped rather than appended to.
        # Before any accumulator is restored: the components in this checkpoint must have been
        # measured under the basis this build implements. Adding today's components onto a total
        # accumulated under another definition would produce a decomposition that sums correctly
        # and describes no Hamiltonian.
        require_compatible_schema(state_of_path, what=f"the checkpoint in {directory}")
        updates_done = int(state_of_path["updates_completed"])
        cumulative = float(state_of_path["cumulative_work_kj_mol"])
        since = float(state_of_path["work_since_last_observation_kj_mol"])
        cumulative_components = ComponentWork.from_mapping(
            state_of_path["cumulative_component_work_kj_mol"])
        since_components = ComponentWork.from_mapping(
            state_of_path["component_work_since_last_observation_kj_mol"])
        # RESTORED AS USEFUL, because that is what they are: those evaluations produced the
        # committed work, observations, frames and state rows this resume is continuing from.
        # They were previously added to `discarded`, which made a resumed path report its own
        # committed work as waste and gave it a different useful total from an identical
        # uninterrupted path -- the exact comparison the counters exist for.
        #
        # What IS lost is whatever the dead process evaluated after this generation committed,
        # and a checkpoint cannot know that number: recording it would need a durable write after
        # every evaluation. So it is marked unobservable rather than guessed at or silently
        # counted as zero.
        restored = EvaluationCounters.from_record(state_of_path.get("evaluation_counters"))
        restored.discarded_is_complete = False
        counters = restored
        probe.counters = counters
        rows_emitted = int(state_of_path["work_rows"])
        frames_emitted = int(state_of_path["frames"])
        state_rows_emitted = int(state_of_path["state_rows"])
        switcher.set_tau(simulation.context, system, taus[updates_done])
        _truncate_csv(directory / OBSERVATIONS_CSV, rows_emitted, OBSERVATION_COLUMNS)
        _truncate_csv(directory / STATE_CSV, state_rows_emitted, STATE_COLUMNS)
        log(f"  path {index:4d}: resuming at step {updates_done * interval} of "
            f"{schedule['switching_steps']} ({rows_emitted} work row(s), {frames_emitted} "
            f"frame(s) kept)")
    else:
        positions, boxes = _read_source_frame(source_path, top, frame, implicit=implicit)
        system_box = boxes
        if system_box is not None:
            simulation.context.setPeriodicBoxVectors(*system_box)
        simulation.context.setPositions(positions)
        # A trajectory carries no velocities, so each path draws fresh Maxwell-Boltzmann momenta
        # at the run temperature with its own recorded seed.
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin,
                                                      int(velocity_seed))
        updates_done = 0
        cumulative = since = 0.0
        cumulative_components = since_components = ComponentWork.zero()
        rows_emitted = frames_emitted = state_rows_emitted = 0
        for path, columns in ((directory / OBSERVATIONS_CSV, OBSERVATION_COLUMNS),
                              (directory / STATE_CSV, STATE_COLUMNS)):
            with path.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=list(columns)).writeheader()
        staged.unlink(missing_ok=True)

    # -- the streams ---------------------------------------------------------------------------
    netcdf = _open_staged_netcdf(staged, frames_emitted)

    def append(path: Path, columns, row: dict[str, Any]) -> None:
        with path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(columns)).writerow(row)

    def write_frame(protocol_step: int, switching_time_ps: float) -> None:
        nonlocal frames_emitted
        # A crash here, after the frame is durable and before the checkpoint that vouches for it,
        # is the case the committed counters exist for: the trajectory is then LONGER than the
        # bookkeeping, and keeping the extra frame would put the path's coordinates permanently
        # ahead of its work rows.
        fault("before-frame")
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        lengths = angles = None
        if not implicit:
            vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            # NOT unpacked as `a, b, c, alpha, beta, gamma`: `beta` is the reciprocal temperature
            # in this module, and binding the box's beta ANGLE to that name once made it local to
            # the writer and put the angle times the work into `cumulative_reduced_work`.
            box = mdtraj.utils.box_vectors_to_lengths_and_angles(
                vectors[0], vectors[1], vectors[2])
            lengths = [box[0] * 10.0, box[1] * 10.0, box[2] * 10.0]
            angles = [box[3], box[4], box[5]]
        netcdf.write(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                     time=switching_time_ps, cell_lengths=lengths, cell_angles=angles)
        netcdf.flush()
        frames_emitted += 1
        fault("after-frame")

    def write_observation(observation_index: int, protocol_step: int, switching_time_ps: float,
                          tau: float, incremental: float, wrote_frame: bool,
                          increment_components) -> None:
        """One observation row: work since the last one, and potentials AT THIS ROW'S COORDINATE.

        The observation probe runs HERE, immediately before the row is written and after the
        frame for this step has been saved, so the coordinate it measures is exactly the
        coordinate `coordinate_frame_index` names. There is no gap in which the Context could
        move between the two.
        """
        nonlocal rows_emitted
        # The potentials first, because they are a measurement and the row is a record of it. If
        # this raises, no row is written -- rather than a row with empty potential cells that
        # looks like a legitimately unaligned observation.
        potentials = observe_potentials(tau, wrote_frame=wrote_frame)

        fault("before-work-row")
        row = {
            "path_index": index,
            "observation_index": observation_index,
            # The index into the path trajectory when this observation also saved a frame, and
            # empty when it did not. The cadences are independent, so a row that claimed a frame
            # index it does not have would be a lie a reader could not detect -- and the potential
            # columns beside it are empty for exactly the same reason.
            "coordinate_frame_index": frames_emitted - 1 if wrote_frame else "",
            "source_frame_index": frame,
            "protocol_step": protocol_step,
            "switching_time_ps": switching_time_ps,
            "tau": tau,
            "incremental_work_kj_mol": incremental,
            "cumulative_work_kj_mol": cumulative,
            "cumulative_reduced_work": beta * cumulative,
            "temperature_kelvin": temperature,
            "integrator_seed": integrator_seed,
            "velocity_seed": velocity_seed,
        }
        row.update(increment_components.row("delta_work"))
        row.update(cumulative_components.row("total_work"))
        row.update(potentials)
        append(directory / OBSERVATIONS_CSV, OBSERVATION_COLUMNS, row)
        rows_emitted += 1
        fault("after-work-row")

    def write_state(protocol_step: int, switching_time_ps: float, tau: float) -> None:
        nonlocal state_rows_emitted
        fault("before-state-row")
        append(directory / STATE_CSV, STATE_COLUMNS,
               _state_row(simulation, index=index, protocol_step=protocol_step,
                          switching_time_ps=switching_time_ps, tau=tau, implicit=implicit,
                          temperature_unit=unit.kelvin))
        state_rows_emitted += 1
        fault("after-state-row")

    def save_checkpoint(updates_completed: int) -> None:
        """One crash-atomic generation: a new checkpoint, its sidecar, then the pointer.

        The committed pair is never overwritten and the pointer is replaced last, so a crash
        anywhere in here leaves the PREVIOUS generation committed and complete. The old design
        overwrote the binary state and then replaced the sidecar, and a crash between those two
        writes left a new Context paired with old accumulated work -- a resume that looks
        successful and is measuring a path that was never run.
        """
        netcdf.flush()
        commit_generation(
            directory,
            write_checkpoint=lambda path: simulation.saveCheckpoint(str(path)),
            state={
                "fingerprint": fingerprint,
                "path_index": index,
                "source_frame_index": frame,
                "updates_completed": updates_completed,
                "protocol_step": updates_completed * interval,
                "tau": taus[updates_completed],
                "cumulative_work_kj_mol": cumulative,
                "work_since_last_observation_kj_mol": since,
                # The component accumulators are committed with the same transaction as the total.
                # Committing them separately would let a resume restore a total from one
                # generation and components from another, and the sum identity would then hold on
                # every row while describing two different paths.
                "decomposition_schema": {"name": DECOMPOSITION_SCHEMA["name"],
                                         "version": DECOMPOSITION_SCHEMA["version"]},
                "cumulative_component_work_kj_mol": cumulative_components.mapping(),
                "component_work_since_last_observation_kj_mol": since_components.mapping(),
                # Counted into the commit, so a resume knows what the interrupted generation had
                # already spent and the cost report does not flatter by omitting it.
                "evaluation_counters": counters.record(),
                "work_rows": rows_emitted,
                "frames": frames_emitted,
                "state_rows": state_rows_emitted,
                "integrator_seed": integrator_seed,
                "velocity_seed": velocity_seed,
                "trajectory": trajectory_name,
            })

    def time_of(step: int) -> float:
        return round(step * float(dynamics["timestep_fs"]) / 1000.0, 9)

    # -- step 0: the source configuration, before any work ---------------------------------------
    if updates_done == 0 and rows_emitted == 0:
        wrote = False
        if 0 % frame_every == 0:
            write_frame(0, 0.0)
            wrote = True
        # Observation zero: the source configuration under the source Hamiltonian, with exactly
        # zero work in every component because nothing has moved yet. Its potentials are measured
        # by the observation probe like any other row's -- at the coordinate it saved.
        write_observation(0, 0, 0.0, taus[0], 0.0, wrote, ComponentWork.zero())
        if state_every:
            write_state(0, 0.0, taus[0])

    # -- the switch ------------------------------------------------------------------------------
    for update in range(updates_done, updates):
        tau_before, tau_after = taus[update], taus[update + 1]

        # THE WORK-BASIS PROBE, at the frozen pre-switch coordinate x_j. This is where work is
        # defined, and it is NOT where this update's observation potentials come from -- those are
        # measured after the propagation below, at the coordinate the row actually saves.
        components, before = measure_components(tau_before, what="the frozen pre-switch coordinate")

        switcher.set_tau(simulation.context, system, tau_after)
        counters.parameter_updates += 1
        after = simulation.context.getState(getEnergy=True).getPotentialEnergy(
            ).value_in_unit(unit.kilojoule_per_mole)
        counters.direct_work_energy_evaluations += 1
        increment = after - before

        # Derived from the fit, INDEPENDENTLY of `increment`. The two are then required to agree:
        # deriving one from the other would make the identity true by construction and it would
        # test nothing.
        increment_components = components.work_between(tau_before, tau_after)
        allowed = reconstruction_tolerance(max(abs(before), abs(after)), precision=precision)
        if abs(increment_components.total - increment) > allowed:
            raise SystemExit(
                f"path {index}, update {update}: the component works do not sum to the measured "
                f"work. Measured {increment:.6f} kJ/mol from U({tau_after}) - U({tau_before}); "
                f"components sum to {increment_components.total:.6f} "
                f"(non_scaled {increment_components.non_scaled:.6f}, "
                f"sqrt_scaled {increment_components.sqrt_scaled:.6f}, "
                f"lin_scaled {increment_components.lin_scaled:.6f}); the difference "
                f"{increment_components.total - increment:.3e} exceeds the {precision}-precision "
                f"tolerance {allowed:.3e}.\n"
                f"  Refusing rather than writing a decomposition of a work value it does not "
                f"reproduce.")

        cumulative += increment
        since += increment
        cumulative_components = cumulative_components + increment_components
        since_components = since_components + increment_components
        simulation.step(interval)

        step = (update + 1) * interval
        wrote = False
        if step % frame_every == 0:
            write_frame(step, time_of(step))
            wrote = True
        if step % observe_every == 0:
            write_observation(step // observe_every, step, time_of(step), tau_after,
                              since, wrote, since_components)
            since = 0.0
            since_components = ComponentWork.zero()
        if state_every and step % state_every == 0:
            write_state(step, time_of(step), tau_after)
        if checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(update + 1)

    # -- finish: flush, validate, publish, and only then declare completion -----------------------
    final = simulation.context.getState(getPositions=True, getVelocities=True,
                                        getParameters=True, enforcePeriodicBox=False)
    (directory / "final_state.xml").write_text(XmlSerializer.serialize(final), encoding="utf-8")
    netcdf.close()

    if rows_emitted != schedule["number_of_observations"]:
        raise SystemExit(f"path {index} wrote {rows_emitted} observations, expected "
                         f"{schedule['number_of_observations']}")
    if frames_emitted != schedule["number_of_frames"]:
        raise SystemExit(f"path {index} wrote {frames_emitted} frames, expected "
                         f"{schedule['number_of_frames']}")
    if state_every and state_rows_emitted != schedule["number_of_state_rows"]:
        raise SystemExit(f"path {index} wrote {state_rows_emitted} state rows, expected "
                         f"{schedule['number_of_state_rows']}")

    with (directory / OBSERVATIONS_CSV).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if float(rows[0]["cumulative_work_kj_mol"]) != 0.0:
        raise SystemExit(f"path {index} observation 0 has non-zero work")
    if float(rows[0]["tau"]) != float(ais["tau_start"]) \
            or float(rows[-1]["tau"]) != float(ais["tau_end"]):
        raise SystemExit(f"path {index} does not span tau_start -> tau_end")

    # The identity on the WHOLE path, checked against what was actually written to the file rather
    # than against the accumulators still in memory. An accumulator that agrees with itself proves
    # nothing about the row a reader will load.
    final_total = float(rows[-1]["cumulative_work_kj_mol"])
    final_components = sum(float(rows[-1][f"total_work_{group}_kj_mol"]) for group in GROUPS)
    allowed = reconstruction_tolerance(final_total, precision=precision) * max(rows_emitted, 1)
    if abs(final_components - final_total) > allowed:
        raise SystemExit(
            f"path {index}: the cumulative component works in {OBSERVATIONS_CSV} sum to "
            f"{final_components:.6f} kJ/mol but the cumulative total is {final_total:.6f}. "
            f"The difference {final_components - final_total:.3e} exceeds {allowed:.3e} "
            f"(per-update {precision} tolerance accumulated over {rows_emitted} rows).")

    # The staged trajectory becomes the published one only now, in one atomic move. Until this
    # point nothing at the run root could be mistaken for a finished path.
    os.replace(staged, published)
    with mdtraj.formats.NetCDFTrajectoryFile(str(published)) as handle:
        published_frames = len(handle)
    if published_frames != schedule["number_of_frames"]:
        raise SystemExit(f"path {index}: {published.name} holds {published_frames} frames, "
                         f"expected {schedule['number_of_frames']}")

    completion = {
        "status": "completed",
        "path_index": index,
        "source_frame_index": frame,
        "observations": rows_emitted,
        "frames": frames_emitted,
        "state_rows": state_rows_emitted,
        "total_work_kj_mol": cumulative,
        "total_reduced_work": beta * cumulative,
        "decomposition_schema": {"name": DECOMPOSITION_SCHEMA["name"],
                                 "version": DECOMPOSITION_SCHEMA["version"]},
        "decomposition_schema_version": DECOMPOSITION_SCHEMA["version"],
        **cumulative_components.row("total_work"),
        # Four counters and their sum, not one number. `switching_energy_evaluations` counted
        # basis probes only and silently omitted the two direct evaluations every switch already
        # performed, so it understated a path's cost by exactly the part that predates the
        # decomposition.
        "evaluation_counters": counters.record(),
        "integrator_seed": integrator_seed,
        "velocity_seed": velocity_seed,
        "trajectory": trajectory_name,
        "mpi_rank": rank,
        "platform": simulation.context.getPlatform().getName(),
        "resumed": bool(state_of_path),
        # What a later invocation checks before believing any of the above.
        "fingerprint": fingerprint,
        "outputs": {
            "trajectory": file_facts(published),
            "observations": file_facts(directory / OBSERVATIONS_CSV),
            # The final state is an artefact this path CLAIMS, so it is hashed with the rest.
            # Leaving it out meant a completed path could be skipped while the one file describing
            # where it ended had been truncated or replaced.
            "final_state": file_facts(directory / "final_state.xml"),
            **({"system_table": file_facts(directory / STATE_CSV)}
               if (directory / STATE_CSV).is_file() else {}),
        },
    }
    # Atomic, and last. Until this file lands the path is incomplete, and a half-written marker
    # would be read as a finished path by the very next invocation.
    from ..openmm.checkpoint import write_durably

    write_durably(marker, (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode())
    # The transaction has served its purpose; leaving it would invite a resume of finished work.
    clear_committed(directory)
    log(f"  path {index:4d}: frame {frame}, {rows_emitted} observations, {frames_emitted} "
        f"frames, W = {cumulative:.4f} kJ/mol (reduced {beta * cumulative:.4f})")
    return completion


def _read_source_frame(source_path: Path, top, frame: int, *, implicit: bool):
    """One frame's positions and box, in the reduced form OpenMM requires."""
    import mdtraj
    from openmm import unit

    positions = boxes = None
    for offset, chunk in enumerate(mdtraj.iterload(str(source_path), top=top, chunk=50)):
        if offset * 50 <= frame < offset * 50 + chunk.n_frames:
            local = frame - offset * 50
            positions = chunk.xyz[local] * unit.nanometer
            if not implicit and chunk.unitcell_lengths is not None:
                # Rebuilt from lengths and angles rather than handed over as the vectors the file
                # happens to store. OpenMM requires REDUCED form, and a truncated octahedron
                # written by any of the usual tools is not in it -- `setPeriodicBoxVectors` then
                # refuses and every path dies at its first frame.
                import numpy
                from openmm.app.internal.unitcell import computePeriodicBoxVectors

                lengths = chunk.unitcell_lengths[local]
                angles = numpy.radians(chunk.unitcell_angles[local])
                boxes = computePeriodicBoxVectors(
                    float(lengths[0]), float(lengths[1]), float(lengths[2]),
                    float(angles[0]), float(angles[1]), float(angles[2]))
            break
    if positions is None:
        raise SystemExit(f"could not read frame {frame} from {source_path}")
    return positions, boxes


def ais_main(run: dict[str, Any], argv: list[str] | None = None) -> int:
    """Run the switching paths this generated script describes."""
    args = ais_parser(run.get("description", "AIS switching paths")).parse_args(argv)

    from openmm import LangevinMiddleIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    from .schedule import switching_schedule
    from ..openmm.timestep import resolve_timestep_fs
    from ..openmm.system import classify_omega_bonds
    from ..md._stages import derive_seed
    from ..rest2 import TauSwitcher
    from ..md.stage import solute_atom_indices

    from ..remd.mpi import Coordination
    from . import path_trajectory_name, paths_for_rank

    ais, source_cfg, dynamics = run["ais"], run["ais_source"], run["dynamics"]
    reporting = run["reporting"]

    # EVERYTHING is validated before `-odir` exists, here in the shared runtime rather than only
    # in `md-openmm md-run`: the generated `AIS.py` calls this function directly. Paths are
    # independent, so `-ng` is simply the worker count, and it must equal the communicator.
    topology_path = Path(args.topology)
    system_path = Path(args.system)
    source_path = Path(args.source or source_cfg["trajectory"] or "")
    out = Path(args.out_dir).resolve()

    from ..run.preflight import PreflightError, preflight_ais

    try:
        checked = preflight_ais(
            topology=topology_path, system=system_path, source=source_path,
            number_of_groups=args.number_of_groups,
            output=args.output or out / "AIS.out", log=args.log or out / "AIS.log",
            cpu=bool(args.cpu),
            device=int(args.device) if args.device is not None else None,
            # The complete resolved configuration, so the schedule, the barostat refusal, the
            # source window, the atom counts, the force classification and the frame selection
            # are all decided BEFORE `-odir` exists -- and so what runs below is the plan that
            # was validated rather than a second one built from the same inputs.
            dynamics=dynamics, ais=ais, reporting=reporting, source_config=source_cfg,
            trajectory=args.trajectory, coordinates=args.coordinates, restart=args.restart,
            checkpoint=args.checkpoint, groupfile=args.groupfile,
            # The identity is decided HERE, before `-odir` exists: the source digest, the
            # fingerprint, the run document, and what this invocation IS -- fresh, resume or
            # overwrite. It used to be settled after the directory and both rank reports had
            # been created, so an incompatible source was refused by a run that had already
            # produced a directory indistinguishable from one that started.
            out_dir=out, resolved_config=run.get("resolved_config"),
            overwrite=bool(args.overwrite), resume=bool(args.resume))
    except PreflightError as refusal:
        print(f"AIS: {refusal}", file=sys.stderr)
        return 2

    coordination = checked.coordination
    rank, size = coordination.rank, coordination.size
    source_format = checked.source_format

    from ..run.preflight import reject_contradictory_continuation

    try:
        reject_contradictory_continuation(resume=bool(args.resume),
                                          overwrite=bool(args.overwrite), what="AIS")
    except PreflightError as refusal:
        print(f"AIS: {refusal}", file=sys.stderr)
        return 2

    if args.check:
        # READ-ONLY, and it returns HERE, before `-odir` is created. `--check` used to make the
        # directory, both reports and `selected_source_frames.csv`, then say nothing was
        # switched -- leaving a tree that a reader, or a later `--resume`, takes for a run that
        # started. The schedule, the source window, the frame selection and the Force audit are
        # all in the preflight result already.
        from ..run.preflight import report_check

        schedule = checked.schedule
        return report_check(
            checked, what="AIS",
            extra=[("source", f"{source_path.name} ({source_format.upper()}, "
                              f"{checked.source_frames} frame(s), "
                              f"{checked.eligible_frames} eligible)"),
                   ("paths", f"{len(checked.chosen_frames)} from frames "
                             f"{list(checked.chosen_frames)[:8]}"),
                   ("switching", f"{schedule['switching_steps']} steps, "
                                 f"{schedule['number_of_updates']} updates, "
                                 f"{schedule['number_of_observations']} observations"),
                   ("ensemble", "fixed volume; no barostat")])

    # OVERWRITE FIRST, before anything is created, and only on rank 0 with every rank agreeing.
    # Clearing after the reports were opened meant `--overwrite` deleted files the same
    # invocation had just written.
    if checked.disposition == "overwrite":
        cleared = None
        if rank == 0:
            try:
                cleared = clear_run_directory(out, identity=checked.identity,
                                              paths=len(checked.chosen_frames),
                                              ranks=coordination.size)
            except BaseException as broken:                 # noqa: BLE001 - reported collectively
                cleared = f"failed: {type(broken).__name__}: {broken}"
        if size > 1:
            for message in coordination.allgather(
                    cleared if isinstance(cleared, str) else None):
                if message:
                    coordination.fail(f"AIS: rank 0 could not clear {out}: {message}")
        elif isinstance(cleared, str):
            print(f"AIS: {cleared}", file=sys.stderr)
            return 2
        coordination.barrier()

    out.mkdir(parents=True, exist_ok=True)
    # Every rank keeps its own pair. An explicitly named -log or -o is suffixed the same way an
    # unnamed one is: without that, N ranks race to rename the same temporary and the run dies
    # with a FileNotFoundError that says nothing about the cause. A rank that failed to bind its
    # device is exactly what a multi-GPU run needs to be able to show, so the other ranks' files
    # are kept beside rank 0's rather than discarded.
    from ..remd.executor import report_path_for_rank

    log_path = Path(report_path_for_rank(str(Path(args.log) if args.log else out / "AIS.log"),
                                         rank))
    out_path = Path(report_path_for_rank(str(Path(args.output) if args.output else out / "AIS.out"),
                                         rank))

    from ..build.simout import SimulationOutput

    sim_out = SimulationOutput(out_path, title=f"AIS: {ais['number_of_paths']} switching paths",
                               log_path=log_path)
    sim_out.heading("Inputs")
    sim_out.field("topology", topology_path)
    sim_out.field("system", system_path)
    sim_out.field("source", f"{source_path} ({source_format.upper()})")
    sim_out.field("output directory", out)

    log = LogWriter(log_path, record_type="md-ais")
    log.update(simulation_output=str(out_path))
    log("md-openmm AIS")
    log("=" * 68)

    try:
        # CONSUMED, not re-derived. Every one of these was computed by `preflight_ais` before
        # `-odir` existed: the pair was deserialised once, the timestep resolved against the
        # masses in THAT System, the schedule built, the barostat refused, the solute and the
        # omega bonds classified and the Force layout audited. Recomputing any of it here would
        # be a second implementation of a policy that already ran, and the one that decides what
        # happens would be this one -- the one nothing refused on.
        pdb, base = checked.loaded.pdb, checked.loaded.system
        timestep = checked.timestep
        dynamics = dict(dynamics, timestep_fs=timestep["timestep_fs"])
        log.field("timestep", f"{timestep['timestep_fs']} fs (requested "
                              f"{timestep['requested']!r}, {timestep['basis']})")
        log.update(timestep=timestep)

        schedule = checked.schedule
        implicit = checked.loaded.implicit
        solute = list(checked.solute)
        omega = checked.notes["omega"]
        excluded = [tuple(int(a) for a in bond) for bond in checked.excluded_bonds]

        log.heading("Path")
        log.field("tau", f"{ais['tau_start']} -> {ais['tau_end']} (linear)")
        log.field("switching", f"{schedule['switching_steps']} steps "
                               f"= {schedule['switching_ps']:g} ps at {dynamics['timestep_fs']} fs")
        log.field("parameter updates", f"{schedule['number_of_updates']} "
                                       f"(every {schedule['parameter_update_interval_steps']} step)")
        log.field("observations", f"{schedule['number_of_observations']} "
                                  f"(work, every {schedule['observation_interval_steps']} steps, "
                                  f"both endpoints included)")
        log.field("frames", f"{schedule['number_of_frames']} "
                            f"(every {schedule['trajectory_interval_steps']} steps)")
        log.field("state rows", f"{schedule['number_of_state_rows']} "
                                f"(every {schedule['state_interval_steps']} steps)"
                  if schedule["state_interval_steps"] else "disabled (system_printout = 0)")
        log.field("checkpoints", f"{schedule['number_of_checkpoints']} "
                                 f"(every {schedule['checkpoint_interval_steps']} steps)"
                  if schedule["checkpoint_interval_steps"] else
                  "disabled (checkpoint_printout = 0): an interrupted path restarts from its "
                  "source frame")
        log.field("observation 0", "the source configuration, before any work")
        log.field("omega bonds", f"{len(excluded)} left unscaled")
        log.field("ensemble", "fixed volume; no barostat")

        # --- the source ensemble -----------------------------------------------------------
        import mdtraj

        top = mdtraj.Topology.from_openmm(pdb.topology)
        # The window, the frame count and the source's own atom count were all established by
        # the preflight, against this same pair, before anything was written.
        n_frames = checked.source_frames
        first = checked.notes["first_frame"]
        last = checked.notes["last_frame"]
        stride = checked.notes["frame_stride"]
        eligible = list(range(first, last + 1, stride))
        source_atoms = checked.source_atoms
        # Hashed ONCE, here, and reused by both the record and the resume fingerprint. A
        # production trajectory is large and is already read frame by frame; digesting it twice
        # would double that for a number that has one value.
        source_facts = checked.source_facts

        log.heading("Source ensemble")
        log.field("trajectory", f"{source_path}  ({source_format.upper()})")
        log.field("atoms", f"{source_atoms}, matching {topology_path.name}"
                  if source_atoms is not None else "not stated by the file format")
        log.field("frames", f"{n_frames} total, {len(eligible)} eligible "
                            f"(frames {first}..{last} inclusive"
                            + (f", every {stride}" if stride > 1 else "") + ")")
        log.field("selection", source_cfg["selection"])
        # Stated, not verified. Nothing in a coordinate trajectory records the Hamiltonian it was
        # sampled under, so this is an assertion the configuration makes and the log repeats --
        # written down precisely so that a reader can check it against the run that produced the
        # file rather than assume it was checked here.
        log.field("asserted ensemble", f"tau = {ais['tau_start']} (ASSERTED by ais.tau_start, "
                                       f"not verifiable from the trajectory itself)")
        log.field("velocities", "not read from the source; each path draws fresh "
                                f"Maxwell-Boltzmann momenta at {dynamics['temperature_K']} K "
                                f"with its own recorded seed")

        chosen = list(checked.chosen_frames)
        log.field("paths", f"{len(chosen)} starting from frames {chosen[:8]}"
                           + (" ..." if len(chosen) > 8 else ""))

        log.update(
            schedule=schedule,
            source={"trajectory": source_path.name, "format": source_format,
                    "atoms": None if source_atoms is None else int(source_atoms),
                    "tau_asserted": float(ais["tau_start"]),
                    "tau_verified_from_file": False,
                    "velocity_policy": "resampled_maxwell_boltzmann",
                    "frames_total": n_frames,
                    "first_frame": first, "last_frame": last,
                    "frame_stride": stride,
                    "selection": source_cfg["selection"],
                    "allow_repeated_frames": bool(source_cfg["allow_repeated_frames"]),
                    "chosen_frames": chosen},
            thermodynamic_states={"source_tau": float(ais["tau_start"]),
                                  "target_tau": float(ais["tau_end"]),
                                  "temperature_K": float(dynamics["temperature_K"]),
                                  "ensemble": "fixed volume (NVT), no barostat"},
            work_convention="delta_W_j = U(tau_{j+1}, x_j) - U(tau_j, x_j); parameters move at "
                            "frozen coordinates, then the configuration propagates",
            inputs={"topology": file_facts(topology_path), "system": file_facts(system_path),
                    "source": source_facts},
            implicit=bool(implicit),
        )

        # `--check` used to return HERE, having written `-odir`, both reports and the
        # selected-frame table. It returns before any of that now -- see the `report_check` call
        # above, which prints from the preflight result and creates nothing -- so there is
        # nothing left to do at this point and this branch is gone rather than left as a second
        # answer to the same question.

        # --- run the paths -------------------------------------------------------------------
        # The preflight already resolved this -- machine settings, device policy, rank placement
        # and a proved Context -- before this function created anything. Consuming its result is
        # what keeps ONE platform policy in the package: resolving again here would be a second
        # implementation, and the one that runs is never the one that gets fixed.
        acceleration = checked.acceleration
        device = checked.device_index
        device_policy = checked.device_policy_detail
        log.field("platform", f"{acceleration.name}"
                              + (f" device {acceleration.device_index}"
                                 if acceleration.device_index is not None else ""))
        log.field("device policy", device_policy)
        log.update(acceleration=checked.record())

        # CONSUMED from the plan. The fingerprint and the identity document were built by the
        # preflight, from the same digests, before this directory existed -- and the decision
        # about whether this directory may be written to at all was made and agreed there.
        # Rebuilding either here would be a second derivation of one string, and the one that
        # decides would be whichever ran later.
        path_fingerprint = checked.fingerprint
        identity_document = checked.identity
        if rank == 0 and not (out / RUN_IDENTITY).is_file():
            write_atomically(out / RUN_IDENTITY,
                             json.dumps(identity_document, indent=2, sort_keys=True) + "\n")
        coordination.barrier()
        log.field("disposition", checked.disposition)
        log.update(run_identity=identity_document, disposition=checked.disposition)

        # AFTER the identity, never before it. `--overwrite` clears this directory, so a
        # selected-frame table written earlier was deleted a moment later and the run then failed
        # hashing a file it had removed itself. The identity is what says whether this directory
        # is this run at all; nothing may be written until it does.
        #
        # Every rank computes the SAME table -- `choose_frames` is a pure function of the window,
        # the count and the run seed -- so only rank 0 writes it, and the others would otherwise
        # race to truncate the file rank 0 is writing. An existing one is left alone:
        # `require_same_run` has already established that the selection is unchanged, so
        # rewriting could only produce the same bytes, and a write that can only be a no-op is a
        # write that can still be interrupted.
        if rank == 0:
            table = out / "selected_source_frames.csv"
            if not table.is_file():
                write_selected_frames(table, chosen, seed=int(dynamics["seed"]))
        coordination.barrier()

        # The switcher the preflight built and audited, not a second one over the same System.
        switcher = checked.switcher
        taus = schedule["taus"]
        interval = schedule["parameter_update_interval_steps"]
        per_observation = schedule["updates_per_observation"]
        temperature = float(dynamics["temperature_K"])
        beta = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * temperature * unit.kelvin).value_in_unit(
            unit.kilojoule_per_mole)

        # Which paths this worker runs. `paths_for_rank` is a pure function of
        # (rank, size, total): a given global path id always owns the same file name, so a
        # restart under a different worker count lands on the same trajectories rather than
        # silently reshuffling which path is which.
        wanted = _selected(args.paths, len(chosen))
        if size > 1:
            mine = set(paths_for_rank(rank, size, len(chosen)))
            wanted = [index for index in wanted if index in mine]
            log.field("mpi", f"rank {rank} of {size}: {len(wanted)} of {len(chosen)} path(s)")
            log.update(mpi={"rank": rank, "size": size, "paths": list(wanted)})

        sim_out.heading("Schedule")
        sim_out.field("tau", f"{ais['tau_start']} -> {ais['tau_end']} (linear)")
        sim_out.field("switching", f"{schedule['switching_steps']} steps "
                                   f"= {schedule['switching_ps']:g} ps")
        sim_out.field("work observations", f"{schedule['number_of_observations']} "
                                           f"(every {schedule['observation_interval_steps']})")
        sim_out.field("trajectory frames", f"{schedule['number_of_frames']} "
                                           f"(every {schedule['trajectory_interval_steps']})")
        sim_out.field("state rows", f"{schedule['number_of_state_rows']} "
                                    f"(every {schedule['state_interval_steps']})"
                      if schedule["state_interval_steps"] else "disabled")
        sim_out.field("checkpoints", f"{schedule['number_of_checkpoints']} "
                                     f"(every {schedule['checkpoint_interval_steps']})"
                      if schedule["checkpoint_interval_steps"] else "disabled")
        sim_out.field("platform", acceleration.name)
        if size > 1:
            sim_out.field("mpi", f"rank {rank} of {size}, {len(wanted)} path(s) of {len(chosen)}")

        sim_out.heading("Paths")
        sim_out.table_header(("path", "frame", "obs", "frames", "W kJ/mol", "reduced"),
                             (6, 8, 6, 8, 14, 12))

        log.heading("Paths")
        completed: list[dict[str, Any]] = []
        for index in wanted:
            record = run_one_path(
                index=index, chosen=chosen, out=out, schedule=schedule, taus=taus,
                switcher=switcher, simulation_inputs=dict(
                    topology=pdb.topology, source_path=source_path, mdtraj_top=top,
                    implicit=implicit, acceleration=acceleration),
                dynamics=dynamics, ais=ais, beta=beta, temperature=temperature,
                rank=rank, resume=bool(args.resume), fingerprint=path_fingerprint, log=log)
            if record is not None:
                completed.append(record)
                sim_out.table_row(
                    (record["path_index"], record["source_frame_index"], record["observations"],
                     record.get("frames", 0), float(record["total_work_kj_mol"]),
                     float(record["total_reduced_work"])), (6, 8, 6, 8, 14, 12))

        # The global work table. Every worker has written a completion record per path it owns;
        # rank 0 waits for all of them and assembles ONE table in global path order. The work
        # distribution is the result of the method, and it has to be readable as a single object
        # rather than as N per-rank fragments the reader is left to concatenate correctly.
        coordination.barrier()
        outputs = {"selected_source_frames": file_facts(out / "selected_source_frames.csv",
                                                        relative_to=out)}
        if rank == 0 and not args.paths:
            table = write_work_table(out, chosen)
            log.field("work table", f"{table['rows']} row(s) from {table['paths']} of "
                                    f"{table['requested']} path(s)")
            log.field("HS table", f"{table['hs_rows']} frame-aligned row(s) in {HS_TABLE}")
            outputs["work_table"] = file_facts(out / WORK_TABLE, relative_to=out)
            outputs["work_summary"] = file_facts(out / WORK_SUMMARY, relative_to=out)
            outputs["hs_table"] = file_facts(out / HS_TABLE, relative_to=out)

        log.heading("Outputs")
        log.field("paths completed", f"{len(completed)} of {len(chosen)}")
        log.update(
            platform=openmm_platform_facts(),
            paths=completed,
            outputs=outputs,
            # What the decomposition cost, measured. Three extra potential-energy evaluations per
            # switching update and no extra integration steps -- stated with the count and the
            # seconds so the claim can be checked against the run rather than believed.
            decomposition={
                **DECOMPOSITION_SCHEMA,
                "evaluation_counters": _summed_counters(completed),
                "reconstruction_tolerance": {
                    "precision": (acceleration.properties or {}).get("Precision", "mixed"),
                    "relative_and_floor_kj_mol": list(RECONSTRUCTION_TOLERANCES.get(
                        str((acceleration.properties or {}).get("Precision", "mixed")).lower(),
                        RECONSTRUCTION_TOLERANCES["mixed"])),
                },
            },
        )
        log.complete()
        log.heading("Summary")
        log(f"  {len(completed)} switching path(s), "
            f"{schedule['number_of_observations']} observations each")
        log("  status: completed")
        sim_out.completed(f"{len(completed)} switching path(s), "
                          f"{schedule['number_of_observations']} work observations and "
                          f"{schedule['number_of_frames']} frames each")
    except BaseException as exc:
        log.fail(f"{type(exc).__name__}: {exc}")
        log.heading("Failure")
        log(f"  {type(exc).__name__}: {exc}")
        log.save()
        sim_out.failed(f"{type(exc).__name__}: {exc}")
        print(f"AIS: {exc}", file=sys.stderr)
        if coordination.size > 1:
            # Returning 1 from one rank is not a failed job, it is a hung one: this rank exits
            # and every other rank walks on to the barrier before the global work table and waits
            # there for a participant that has already gone. `fail` prints the reason and takes
            # the communicator down, so the launcher returns promptly and no rank survives.
            coordination.fail(f"AIS rank {rank}: {type(exc).__name__}: {exc}", code=1)
        return 1

    log.save()
    return 0


def run_generated_ais(script: str | Path, argv: list[str] | None = None) -> int:
    """Run the AIS paths described by the `resolved.config` beside this script.

    The whole body of a generated `AIS.py`:

        from md_tools.ais import run_generated_ais
        raise SystemExit(run_generated_ais(__file__))
    """
    from ..build.md import resolve_md_config
    from ..md.stage import resolved_config_beside

    config_path = resolved_config_beside(script)
    resolved = resolve_md_config(config_path)
    if resolved["protocol"] != "AIS":
        raise SystemExit(
            f"{config_path} declares protocol {resolved['protocol']!r}, but this is an AIS "
            f"script. The directory holds a script and a configuration that describe different "
            f"runs; regenerate it.")
    run = {
        "protocol": "AIS",
        "description": f"AIS: {resolved['ais']['number_of_paths']} switching paths",
        "ais": dict(resolved["ais"]),
        "ais_source": dict(resolved["ais_source"]),
        "dynamics": dict(resolved["dynamics"]),
        # The reporting block reaches the runtime. It used to be validated by `build-md` and then
        # left behind, so `system_printout` and `checkpoint_printout` were settings a person wrote
        # that nothing ever read.
        "reporting": dict(resolved["reporting"]),
        "resolved_config": str(config_path),
    }
    return ais_main(run, argv)
