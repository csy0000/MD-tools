"""Project a resolved MD configuration into an inspectable staged project.

This is the implementation behind ``MD_input_gen.py``. It is a PROJECTION layer, not a second
configuration engine: every scientific value comes from the canonical ``SimulationSpec``, and each
stage JSON is a view of that one resolved model. Two places where a default could live is how two
places come to disagree.

The generated project makes explicit what the single-command pipeline left implicit:

* each stage owns a directory, its configuration, its launcher, and (once run) its outputs;
* each stage names the topology and the input state it consumes, and which stage produced them;
* the launcher is a readable artifact -- the command is committed, not recalled from shell history.

Stage order and hand-off:

    min      <- the system bundle's initial_state.xml   (never integrated)
    eq_nvt   <- min/min_final_state.xml
    eq_npt   <- eq_nvt/eq_nvt_final_state.xml
    cMD_1    <- eq_npt/eq_npt_final_state.xml
    REST2_1  <- cMD_1/cMD_1_final_state.xml

A State, not a PDB, is what carries between stages: positions alone would silently discard
velocities and box vectors, restarting the thermostat and the barostat at every boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = ["generate_project", "RUN_MANIFEST_SCHEMA_VERSION", "STAGE_ORDER",
           "IMPLICIT_STAGE_ORDER", "MD_STAGE_ORDER", "IMPLICIT_MD_STAGE_ORDER",
           "stage_order_for"]

RUN_MANIFEST_SCHEMA_VERSION = 1

#: Stage directory names. Conventional MD is `cMD_N` and replica exchange is `REST2_N`, capitalised
#: as the methods are normally written.
#:
#: The two NPT stages are the point of the numbering, not decoration:
#:
#:     eq_npt_1   POSITION-RESTRAINED NPT -- the box relaxes while the solute is held
#:     eq_npt_2   FREE NPT                -- restraints released, the solute relaxes in the
#:                                           equilibrated box
#:
#: Collapsing them into one stage means either releasing the restraint while the density is still
#: settling, or entering production with the solute still held. Both are the kind of error that
#: produces a run that completes and is wrong.
#: The independent random streams a single stage owns. Separate seeds because an integrator and a
#: barostat sharing one would be correlated in a way nobody would think to look for, and the
#: velocity draw must be reproducible on its own so "initialised once" is checkable.
_STAGE_SEED_ROLES = ("integrator", "barostat", "velocity")

STAGE_ORDER = ("min", "eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1", "REST2_1")

#: The implicit-solvent graph. Restrained minimisation, then production.
#:
#: There is no NPT stage and there cannot be one: implicit solvent has no box, so there is no volume
#: to equilibrate and pressure is undefined.
#:
#: Equilibration still happens -- the solute has to settle under restraints before production -- but
#: it is a single stage named `eq` rather than `eq_nvt`. "NVT" names an ensemble at constant volume,
#: and there is no volume here, so the label would describe something the run does not have. The
#: dynamics are the same: constant temperature, solute restrained, no barostat.
IMPLICIT_STAGE_ORDER = ("min", "eq", "cMD_1", "REST2_1")

#: Conventional MD ends at cMD_1. A REST2 stage is not appended to an MD-only project, and no REST2
#: object -- tau ladder, exchange schedule, omega policy -- is constructed for it. Generating one
#: anyway would put a replica-exchange calculation in a project that never asked for it, and its
#: presence in the manifest would misdescribe the run.
MD_STAGE_ORDER = ("min", "eq_nvt", "eq_npt_1", "eq_npt_2", "cMD_1")
IMPLICIT_MD_STAGE_ORDER = ("min", "eq", "cMD_1")

#: Stages that carry the positional restraint on the solute. `eq_npt_2` deliberately does not.
RESTRAINED_STAGES = ("min", "eq", "eq_nvt", "eq_npt_1")

#: Stages that carry a barostat. Empty under implicit solvent, enforced rather than assumed.
#: REST2_1 belongs here: explicit REST2 is NPT replica exchange at one physical pressure, so
#: every replica carries its own independently seeded barostat and its box fluctuates during
#: production. Its absence meant explicit REST2 production ran at fixed volume while the
#: configuration said otherwise -- the box stopped moving at the exact point the science
#: started. Safe to add only now that the exchange criterion carries the pV term.
BAROSTAT_STAGES = ("eq_npt_1", "eq_npt_2", "cMD_1", "REST2_1")


def stage_order_for(solvation_mode: str, method: str = "rest2") -> tuple:
    """The stage graph a solvation mode and production method imply.

    Four graphs, from two independent facts: implicit solvent has no NPT stage, and conventional MD
    has no REST2 stage. Neither is a subset of the other, so both are decided here rather than by
    trimming a single canonical list somewhere downstream.
    """
    from .solvation_mode import IMPLICIT

    implicit = solvation_mode == IMPLICIT
    if method == "md":
        return IMPLICIT_MD_STAGE_ORDER if implicit else MD_STAGE_ORDER
    return IMPLICIT_STAGE_ORDER if implicit else STAGE_ORDER

#: Keys md_config.json may carry that the canonical model does not model. See _resolved_spec.
GENERATOR_ONLY_KEYS = ("conventional_md", "minimization")


def _sha256(path: Path) -> str:
    from .hashing import sha256_file

    return sha256_file(path)


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _write_script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _verify_bundle(manifest_path: Path) -> dict:
    """Read the system manifest and verify every checksum before anything is generated.

    A protocol generated against a tampered or truncated bundle would run and produce numbers.
    """
    manifest = json.loads(manifest_path.read_text())
    version = manifest.get("schema_version")
    if version != 1:
        raise ValueError(
            f"system_manifest.json schema_version is {version!r}, this generator understands 1. "
            "Regenerate the system bundle with the current MD_system_gen.py rather than reading an "
            "old layout as if it were current."
        )
    bundle = manifest_path.parent
    checksums = json.loads((bundle / "checksums.json").read_text())["files"]
    bad = []
    for name, expected in sorted(checksums.items()):
        target = bundle / name
        if not target.is_file():
            bad.append(f"{name}: missing")
        elif _sha256(target) != expected:
            bad.append(f"{name}: checksum mismatch")
    if bad:
        raise ValueError(
            "the system bundle does not match its own checksums, refusing to generate a protocol "
            "against it:\n  " + "\n  ".join(bad)
        )
    return manifest


def _resolved_spec(md_config: dict, manifest: dict):
    """Resolve md_config.json through the canonical typed model.

    The system's identity comes from the BUNDLE, not from md_config: a protocol that could restate
    the molecule would be able to disagree with the system it runs against.
    """
    from .spec import resolve

    document = json.loads(json.dumps(md_config))

    # Generator-level keys: consumed by the STAGED LAYOUT, not by the canonical simulation model.
    # They are removed before resolution because the canonical model rejects unknown fields -- and
    # rightly so. They are recorded in the run manifest with their values so nothing is silent.
    #
    # This is a known seam, documented in the journal: the canonical model describes ONE production
    # method, while a staged protocol has both a cMD stage and a REST2 stage. Until the model grows
    # a multi-stage production block, the cMD duration and the restraint constant live here.
    for key in GENERATOR_ONLY_KEYS:
        document.pop(key, None)

    chemistry_keys = {"forcefield", "solvation", "system_build"}
    intruders = sorted(chemistry_keys & set(document))
    if intruders:
        raise ValueError(
            f"md_config.json contains system-building settings: {', '.join(intruders)}. "
            "Those belong to system_config.json and are already fixed in the prepared bundle; "
            "restating them here would let a protocol silently contradict the system it runs."
        )

    system_block = manifest["system"]
    document.setdefault("system", {})
    document["system"].update({
        "system_id": system_block["id"],
        "route": "smiles" if system_block["input_format"] == "smi" else "pdb",
    })
    if document["system"]["route"] == "smiles":
        document["system"]["smiles"] = system_block.get("smiles")
    else:
        document["system"]["pdb"] = system_block["input_file"]

    # build settings come from the bundle; the model still needs them to validate
    forcefield = json.loads((Path(manifest["_bundle_dir"]) / "forcefield.json").read_text())
    document.setdefault("build", {})
    return resolve.resolve_spec(document), forcefield


def _interpreter_defaults() -> tuple[str, str]:
    """The interpreter that generated this project, and the PYTHONPATH it needed.

    A generated project must say how to run itself. Defaulting to bare `python` is not good enough
    here: the stack's activation script puts AmberTools' interpreter first, and that one has neither
    openmm nor md_templates, so a launcher that trusts `python` fails on the first stage with a
    ModuleNotFoundError that tells the user nothing about which interpreter to use instead.
    """
    import md_templates

    interpreter = sys.executable
    package_root = Path(md_templates.__file__).resolve().parent.parent
    # only needed when running from a source checkout rather than an installed wheel
    needs_path = "site-packages" not in str(package_root)
    return interpreter, (str(package_root) if needs_path else "")


def _stage_launcher(stage: str, interpreter: str, pythonpath: str) -> str:
    """A readable launcher. It calls a package module; it does not reimplement any physics."""
    # APPENDED, not defaulted. `${PYTHONPATH:=...}` keeps whatever the caller already exported, and
    # a caller with a RELATIVE PYTHONPATH (a bare `src`, say) then breaks the moment this script
    # changes directory -- the recorded absolute path is right there and gets ignored. Appending
    # keeps the caller's entries ahead of ours, so they can still shadow the package deliberately,
    # while guaranteeing the interpreter can find it at all.
    path_line = (f'PYTHONPATH="${{PYTHONPATH:+$PYTHONPATH:}}{pythonpath}"\nexport PYTHONPATH\n'
                 if pythonpath else "")
    return f"""#!/usr/bin/env bash
# Stage: {stage}
#
# This script is the record of how this stage is run. It calls the package's stage command; the
# physics lives there, not here, so there is exactly one implementation to audit.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

# The interpreter this project was GENERATED with. Overridable, but it is the one known to have
# both openmm and md_templates -- the stack's activation script puts a different python first.
: "${{PYTHON:={interpreter}}}"
{path_line}: "${{MD_DEVICES:=}}"

if ! "$PYTHON" -c 'import md_templates, openmm' 2>/dev/null; then
    echo "[{stage}] $PYTHON cannot import md_templates and openmm." >&2
    echo "  Set PYTHON to an interpreter that has both, e.g." >&2
    echo "    PYTHON={interpreter} ./{stage}.sh" >&2
    exit 2
fi

cd "$HERE"
echo "[{stage}] start $(date -Is)" | tee -a "$PROJECT/run.log"

"$PYTHON" -m md_templates.openmm.stage --config "{stage}.json" ${{MD_DEVICES:+--devices "$MD_DEVICES"}}

echo "[{stage}] done  $(date -Is)" | tee -a "$PROJECT/run.log"
"""


def _run_all(stages: tuple[str, ...]) -> str:
    # A REST2 stage is invoked once per SEGMENT: each invocation continues the same run, because
    # the runner reads its own committed-generation record to find where the last one stopped.
    # That is why the segment count is a loop here and not a field in the stage JSON -- asking for
    # more sampling must not change the configuration hash.
    def _line(stage: str) -> str:
        if stage.startswith("REST2"):
            return f'run_segments {stage} "$REST2_NUMBER_OF_SEGMENTS"'
        if stage.startswith("cMD"):
            return f'run_segments {stage} "$CMD_NUMBER_OF_SEGMENTS"'
        return f"run_stage {stage}"

    lines = "\n".join(_line(s) for s in stages)
    return f"""#!/usr/bin/env bash
# Run every stage in order. Each stage's own launcher owns how that stage runs; this file owns
# only the ORDER and the record of what was executed.
set -euo pipefail
PROJECT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$PROJECT"

# How many segments of each production stage to run. These are EXECUTION choices and live here,
# never in the scientific JSON: asking for a longer run must not change the configuration hash.
# They are separate because the two stages are separately extensible -- adding cMD segments and
# adding REST2 segments are different requests, and one number could not express both.
: "${{CMD_NUMBER_OF_SEGMENTS:=1}}"
: "${{REST2_NUMBER_OF_SEGMENTS:=2}}"
export CMD_NUMBER_OF_SEGMENTS REST2_NUMBER_OF_SEGMENTS

run_stage () {{
    local s="$1"
    if [ ! -x "$s/$s.sh" ]; then
        echo "missing launcher: $s/$s.sh" >&2; exit 1
    fi
    echo "=== $s ===" | tee -a run.log
    "./$s/$s.sh"
}}

# Run a production stage for N segments. Re-invoking the SAME launcher is what continues the chain:
# both cMD and REST2 read their own committed-generation record to find the restart point, so this
# loop never has to know where the last segment stopped.
#
# Equilibration above runs once. To ADD segments later, invoke the stage launcher directly rather
# than re-running this script -- that continues the same run without re-running minimisation.
run_segments () {{
    local s="$1" n="$2" i
    for (( i=1; i<=n; i++ )); do
        echo "=== $s segment $i/$n ===" | tee -a run.log
        "./$s/$s.sh"
    done
}}

{lines}

echo "all stages complete $(date -Is)" | tee -a run.log
"""


def resolve_inheritance(inherit: Optional[str]) -> Optional[dict]:
    """Parse ``--inherit <run_manifest.json>[:<stage>]`` into a resolved hand-off.

    With a stage, this project SKIPS every stage up to and including it and starts from that
    stage's endpoint State in the other project. That is what makes it worth having: an
    equilibrated system can back many production protocols without being re-equilibrated.

    The cost is that the project is no longer self-contained -- it depends on a file in another
    directory -- so the borrowed state's path, producing stage and sha256 are all recorded. A
    reader must be able to tell which coordinates a run actually started from.

    This is NOT a checkpoint resume. Resuming continues the SAME run inside its own directory
    through the committed-generation record; inheriting starts a NEW run from another run's
    endpoint.
    """
    if inherit is None:
        return None
    text = str(inherit)
    stage = None
    # a Windows-style drive letter is not a concern here, but a path may contain ':' -- split from
    # the right and only accept the tail as a stage if it names one we generate
    if ":" in text:
        head, tail = text.rsplit(":", 1)
        if tail in STAGE_ORDER:
            text, stage = head, tail
    manifest_path = Path(text).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"--inherit manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    resolved = {
        "inherited_from": str(manifest_path),
        "inherited_run_manifest_sha256": _sha256(manifest_path),
        "note": ("lineage only. This is NOT a checkpoint resume: continuing a REST2 run happens in "
                 "that run's own directory through the committed-generation contract."),
    }
    if stage is None:
        return resolved

    project = manifest_path.parent
    state = project / stage / f"{stage}_final_state.xml"
    if not state.is_file():
        raise ValueError(
            f"--inherit names stage {stage!r}, but {state} does not exist. That stage has not been "
            "run in the source project, so there is no endpoint to inherit. Run it there first, or "
            "inherit an earlier stage."
        )
    resolved.update({
        "inherited_stage": stage,
        "inherited_state": str(state),
        "inherited_state_sha256": _sha256(state),
        "skipped_stages": list(STAGE_ORDER[:STAGE_ORDER.index(stage) + 1]),
        "self_contained": False,
        "hand_off_note": (
            f"stages up to and including {stage} were NOT generated; this project starts from the "
            "source project's endpoint State. The state's sha256 is recorded so a changed or "
            "re-run source is detectable."
        ),
    })
    return resolved


def _refuse_keeping_results_from_another_protocol(outdir: Path, hashes: dict) -> None:
    """The guard that makes `--overwrite-generated` safe rather than merely convenient.

    Keeping results while rewriting the stage files is exactly right when the generator changed --
    a fixed launcher, a new stage projection -- and the protocol did not. It is exactly wrong when
    the protocol changed, because the surviving results were produced under the old one and nothing
    in the directory would say so.

    So the protocol hash recorded by the previous generation is compared with this one, and a
    mismatch is refused with the two correct alternatives: discard the results, or start a new
    project that inherits an endpoint from this one.
    """
    from .destination import DestinationExists

    previous = Path(outdir) / "run_manifest.json"
    if not previous.is_file():
        return
    try:
        recorded = json.loads(previous.read_text()).get("configuration_hashes", {})
    except json.JSONDecodeError:
        return
    was, now = recorded.get("protocol_sha256"), hashes.get("protocol_sha256")
    if not was or not now or was == now:
        return
    raise DestinationExists(
        f"refusing to keep the results in {outdir}: they were produced under a different "
        f"protocol.\n"
        f"    recorded : {was[:16]}\n"
        f"    this run : {now[:16]}\n\n"
        "  --overwrite-generated keeps whatever the stages produced, which is correct when the\n"
        "  GENERATOR changed and the protocol did not. Here the protocol itself changed, so those\n"
        "  results would sit beside stage files that no longer describe them.\n\n"
        "  Use --overwrite to discard them, or generate a new project with\n"
        "  --inherit <this project>/run_manifest.json:<stage> to reuse an endpoint without\n"
        "  pretending the old results belong to the new protocol."
    )


def _reporting_for_stage(stage: str, steps: int, full_steps: int, selected_steps: int) -> dict:
    """Which reporting streams this stage can honestly declare.

    An interval longer than the stage produces no frames. Declaring it anyway is the defect this
    repository already fixed once in the other direction -- a manifest describing files that do not
    exist -- so the generator omits what cannot be written and says why.

    Equilibration keeps its state log wherever the cadence fits, because temperature and density
    settling is exactly what a reader wants from those stages; it is the trajectories that are a
    production concern.
    """
    reporting: dict = {"selected_atoms": {"type": "solute"}}
    omitted: dict = {}

    for key, interval, label in (("full_system_interval_steps", full_steps, "all-atom trajectory"),
                                 ("selected_atoms_interval_steps", selected_steps,
                                  "selected-atom trajectory")):
        if steps and interval and interval <= steps:
            reporting[key] = interval
        elif steps:
            omitted[label] = (f"interval {interval:,} steps exceeds this stage's {steps:,} steps")

    # the state log falls back to a cadence that fits, so equilibration is still observable
    if steps:
        state_interval = full_steps if full_steps <= steps else max(1, steps // 10)
        reporting["state_interval_steps"] = state_interval
        if state_interval != full_steps:
            omitted["state log cadence"] = (
                f"{full_steps:,} steps exceeds this stage, so it logs every "
                f"{state_interval:,} steps instead")
    if omitted:
        reporting["omitted"] = omitted
    return reporting


def md_config_method(md_config: dict) -> str:
    """The production method a document asks for, before the spec is resolved.

    The stage graph has to be known before resolution, because which stages exist decides which
    seeds are derived. Read directly and defensively: an unrecognised value falls through to the
    resolver, which reports it properly.
    """
    method = ((md_config.get("protocol") or {}).get("production") or {}).get("method")
    return str(method) if method in ("md", "rest2") else "rest2"


def _read_manifest_solvation(system_manifest) -> dict:
    """The solvation block the prepared bundle recorded.

    Read from the manifest rather than from md_config.json: solvation is a property of the SYSTEM,
    decided when it was built. A protocol cannot change it, and a protocol that appeared to would be
    describing a different System than the one it points at.
    """
    document = json.loads(Path(system_manifest).read_text())
    return (document.get("resolved_system_config") or {}).get("solvation") or {}


def generate_project(*, system_manifest: Path, md_config: dict, outdir: Path,
                     inherit: Optional[str] = None, overwrite: bool = False,
                     dry_run: bool = False) -> dict:
    """Generate the staged project. Transactional; writes nothing on failure."""
    from .segments import (plan_segment_from_duration_and_exchanges, reporting_interval_steps, steps_for_duration)

    inheritance = resolve_inheritance(inherit)
    skipped = set(inheritance.get("skipped_stages", [])) if inheritance else set()
    solvation_mode = str(
        ((_read_manifest_solvation(system_manifest)) or {}).get("mode", "explicit"))
    graph = stage_order_for(solvation_mode, md_config_method(md_config))
    stages_to_generate = tuple(s for s in graph if s not in skipped)
    if not stages_to_generate:
        raise ValueError(
            f"--inherit {inherit!r} would skip every stage, leaving nothing to generate. "
            "Inherit an earlier stage."
        )

    # Checked BEFORE any work, and against the stages this run will actually generate: inheriting
    # from a stage means fewer stage directories, so a fixed list would report collisions for
    # directories that were never going to be written.
    from .destination import (OVERWRITE_GENERATED, check_destination, project_targets, publish,
                              resolve_mode)
    from .seeds import DEFAULT_MASTER_SEED, seed_map, stage_purpose
    check_destination(outdir, project_targets(stages_to_generate),
                      overwrite=overwrite, what="project")

    manifest = _verify_bundle(system_manifest)
    bundle_dir = system_manifest.parent
    manifest["_bundle_dir"] = str(bundle_dir)

    resolution, forcefield = _resolved_spec(md_config, manifest)

    # One master seed for the project, and every stream derived from it by the shared algorithm.
    master_seed = int((md_config.get("randomness") or {}).get("master_seed", DEFAULT_MASTER_SEED))
    seed_records = seed_map(
        master_seed,
        [stage_purpose(stage, role)
         for stage in stages_to_generate for role in _STAGE_SEED_ROLES],
    )
    seed_records["source"] = ("md_config.randomness.master_seed"
                              if (md_config.get("randomness") or {}).get("master_seed") is not None
                              else "default")
    if resolve_mode(overwrite) == OVERWRITE_GENERATED:
        _refuse_keeping_results_from_another_protocol(outdir, resolution["hashes"])
    spec = resolution["spec"]
    integrator = spec.protocol.integrator
    dt = integrator.timestep

    equilibration = spec.protocol.equilibration
    production = spec.protocol.production

    # ---- exact integer step counts for every stage, or a refusal -------------------------------
    summary: list[str] = []
    stage_steps: dict[str, int] = {}
    if equilibration.nvt is not None:
        stage_steps["eq_nvt"] = steps_for_duration(
            equilibration.nvt.value, dt.value, duration_source=equilibration.nvt.source,
            timestep_source=dt.source, duration_label="equilibration.nvt")
    if equilibration.restrained is not None:
        # Implicit solvent's single equilibration stage: restrained constant-temperature dynamics,
        # no barostat, no ensemble label that implies a volume.
        stage_steps["eq"] = steps_for_duration(
            equilibration.restrained.value, dt.value,
            duration_source=equilibration.restrained.source,
            timestep_source=dt.source, duration_label="equilibration.restrained")
    if equilibration.npt is not None:
        # restrained NPT: the box relaxes while the solute is held
        stage_steps["eq_npt_1"] = steps_for_duration(
            equilibration.npt.value, dt.value, duration_source=equilibration.npt.source,
            timestep_source=dt.source, duration_label="equilibration.npt")
    if equilibration.npt_free is not None:
        # free NPT: restraints released, the solute relaxes in the equilibrated box
        stage_steps["eq_npt_2"] = steps_for_duration(
            equilibration.npt_free.value, dt.value,
            duration_source=equilibration.npt_free.source,
            timestep_source=dt.source, duration_label="equilibration.npt_free")

    if production.method == "rest2":
        plan = plan_segment_from_duration_and_exchanges(
            production.duration_per_segment.value,
            production.exchange.number_of_exchanges_per_segment, dt.value,
            duration_source=production.duration_per_segment.source,
            timestep_source=dt.source)
        stage_steps["REST2_1"] = plan.steps_per_segment
        summary.append(
            f"REST2_1  {production.n_replicas} replicas, "
            f"{production.duration_per_segment.source} / "
            f"{production.exchange.number_of_exchanges_per_segment} exchanges = "
            f"{plan.steps_per_segment:,} steps per segment "
            f"({plan.steps_per_exchange:,} per exchange round)")
    elif production.method != "md":
        raise ValueError(
            "md_config.json declares protocol.production.method "
            f"{production.method!r}; implemented methods are 'md' and 'rest2'."
        )

    # The cMD segment length. For an MD-only project this IS the canonical
    # `production.duration_per_segment`; for the REST2 chain, cMD is a pre-production stage whose
    # length comes from the generator-only `conventional_md.duration`. Two fields must never
    # compete for one stage, so which one applies is decided by the method and recorded with its
    # source rather than resolved by precedence.
    stated_cmd = (md_config.get("conventional_md") or {}).get("duration")
    if production.method == "md":
        if stated_cmd is not None:
            raise ValueError(
                "this is an MD-only project, so the cMD segment length is "
                "protocol.production.duration_per_segment, but the document also states "
                f"conventional_md.duration = {stated_cmd!r}.\n"
                "  Two fields would be competing for one stage. Remove conventional_md.duration; it "
                "applies only to the\n  pre-production cMD stage of a REST2 chain."
            )
        cmd_duration = production.duration_per_segment
        cmd_source = "protocol.production.duration_per_segment"
    else:
        from .spec.units import parse_quantity
        cmd_duration = parse_quantity(stated_cmd if stated_cmd is not None else "1 ns",
                                      dimension="time")
        cmd_source = ("conventional_md.duration" if stated_cmd is not None
                      else "default: conventional_md.duration")

    stage_steps["cMD_1"] = steps_for_duration(
        cmd_duration.value, dt.value, duration_source=cmd_duration.source,
        timestep_source=dt.source, duration_label=cmd_source)
    stage_steps["min"] = 0

    reporting = spec.execution.reporting
    full_steps = reporting_interval_steps(
        reporting.all_atom.value, dt.value, interval_source=reporting.all_atom.source,
        timestep_source=dt.source, label="reporting.all_atom")
    selected_steps = reporting_interval_steps(
        reporting.solute.value, dt.value, interval_source=reporting.solute.source,
        timestep_source=dt.source, label="reporting.solute")

    summary.append(f"min      max {equilibration.minimize_max_iterations} iterations")
    if "eq_nvt" in graph:
        summary.append(
            f"eq_nvt   {stage_steps.get('eq_nvt', 0):,} steps ({equilibration.nvt.source})"
            if equilibration.nvt else "eq_nvt   (not configured)")
    # The NPT stages are reported only when the graph has them. Saying "not configured" for a
    # stage that cannot exist under implicit solvent reads as an omission the user could fix.
    if "eq_npt_1" in graph:
        summary.append(f"eq_npt_1 {stage_steps.get('eq_npt_1', 0):,} steps "
                       f"({equilibration.npt.source}, restrained)"
                       if equilibration.npt else "eq_npt_1 (not configured)")
    if "eq_npt_2" in graph:
        summary.append(f"eq_npt_2 {stage_steps.get('eq_npt_2', 0):,} steps "
                       f"({equilibration.npt_free.source}, free)"
                       if equilibration.npt_free else "eq_npt_2 (not configured)")
    if "eq" in graph:
        summary.append(
            f"eq       {stage_steps.get('eq', 0):,} steps "
            f"({equilibration.restrained.source}, restrained; no ensemble label because implicit "
            "solvent has no volume)")
    summary.append(f"cMD_1    {stage_steps['cMD_1']:,} steps ({cmd_duration.source})")
    summary.append(f"reporting: full system every {full_steps:,} steps, "
                   f"selected atoms every {selected_steps:,} steps")

    if dry_run:
        if inheritance and inheritance.get("skipped_stages"):
            summary.append(f"inherited: {inheritance['inherited_stage']} endpoint from "
                           f"{inheritance['inherited_from']}")
            summary.append(f"skipped  : {', '.join(inheritance['skipped_stages'])}")
        return {"system_id": manifest["system"]["id"],
                "n_solute_atoms": manifest["composition"]["n_solute_atoms"],
                "stages": list(stages_to_generate), "summary": summary}

    # ---- transactional generation ---------------------------------------------------------------
    interpreter, pythonpath = _interpreter_defaults()

    staging = outdir.parent / f".{outdir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        inputs = staging / "inputs"
        inputs.mkdir()
        for name in ("system.xml", "topology.pdb", "topology.cif", "initial_state.xml",
                     "forcefield.json", "system_manifest.json", "system.yaml",
                     "checksums.json"):
            shutil.copy2(bundle_dir / name, inputs / name)
        # The simbox record build_simbox produced. Not part of the stage contract, but the REST2
        # stage needs it to assemble the package-format bundle the runner consumes -- copying it
        # here keeps the project self-contained rather than reaching back into the system bundle.
        for extra in bundle_dir.glob("*simbox.json"):
            shutil.copy2(extra, inputs / extra.name)
        # the exact molecular input, so the project is self-contained and the legacy system
        # manifest's `input.pdb` resolves wherever the bundle is assembled
        original = bundle_dir / "original_inputs"
        if original.is_dir():
            shutil.copytree(original, inputs / "original_inputs", dirs_exist_ok=True)

        restraint = (md_config.get("minimization") or {}).get("restraint") or {}
        restraint_k = restraint.get("force_constant_kcal_per_mol_angstrom2", 1.0)

        previous_state = "inputs/initial_state.xml"
        external_producer = None
        if inheritance and inheritance.get("inherited_state"):
            # the first generated stage consumes the OTHER project's endpoint
            previous_state = os.path.relpath(inheritance["inherited_state"], staging)
            external_producer = (f"{Path(inheritance['inherited_from']).parent.name}:"
                                 f"{inheritance['inherited_stage']} (external)")
        stage_records = []
        for stage in stages_to_generate:
            d = staging / stage
            d.mkdir()
            restrained = stage in RESTRAINED_STAGES
            payload = {
                "stage": stage,
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                # Resolved here, once, and carried with the stage. The runner reads these and
                # derives nothing: a seed invented at execution time is a seed no record explains.
                "seeds": {role: seed_records["seeds"][stage_purpose(stage, role)]
                          for role in _STAGE_SEED_ROLES},
                # every stage NAMES what it consumes and which stage produced it
                "input": {
                    "system_xml": "../inputs/system.xml",
                    "topology": "../inputs/topology.pdb",
                    "state": f"../{previous_state}" if previous_state.startswith("inputs/")
                             else f"../{previous_state}",
                    "produced_by": (external_producer if external_producer is not None
                                    else "MD_system_gen.py" if previous_state.startswith("inputs/")
                                    else previous_state.split("/")[0]),
                },
                "output": {
                    "final_state": f"{stage}_final_state.xml",
                    "final_structure": f"{stage}_final.pdb",
                    "checkpoint": f"{stage}.chk",
                    "log": f"{stage}.log",
                    "results": f"{stage}_results.json",
                    # Named here so the manifest and the execution path cannot disagree about which
                    # files exist. A stage that takes no steps writes neither.
                    "trajectory_all_atoms": f"{stage}_all_atoms.dcd",
                    "trajectory_selected_atoms": f"{stage}_selected_atoms.dcd",
                },
                "integrator": {
                    "type": integrator.kind, "timestep": dt.source,
                    "temperature": integrator.temperature.source,
                    "friction": integrator.friction.source,
                },
                "steps": stage_steps.get(stage, 0),
                "restraint": ({
                    "selection": {"type": "solute"},
                    "force_constant_kcal_per_mol_angstrom2": restraint_k,
                    "reference": "inputs/initial_state.xml",
                    "note": "positional restraints on the solute; the reference coordinates are the "
                            "prepared system's, not the previous stage's, so the restraint means "
                            "the same thing in every restrained stage",
                } if restrained else None),
                "reporting": {
                    # A stream is DECLARED only where it can actually produce a frame. The
                    # reporting cadence is a production cadence -- 100 ps between all-atom frames --
                    # and an equilibration stage is 10 ps long, so declaring it there would promise
                    # a trajectory that cannot exist. Rather than attach a reporter that writes
                    # nothing, the interval is omitted and the reason is recorded, which keeps
                    # "everything declared is written" true for every stage.
                    **_reporting_for_stage(stage, stage_steps.get(stage, 0),
                                           full_steps, selected_steps),
                    "selected_atoms": {"type": "solute"},
                    # What the prepared bundle says the selection should resolve to. The stage
                    # re-resolves it against the topology and refuses on a mismatch, so a project
                    # pointed at a different bundle fails instead of writing a trajectory whose
                    # atom order silently means something else.
                    "selected_atoms_expected_count": int(
                        manifest["composition"]["n_solute_atoms"]),
                },
                "execution": {
                    "platform": spec.execution.platform,
                    "precision": spec.execution.precision,
                },
            }
            if stage == "min":
                payload["max_iterations"] = equilibration.minimize_max_iterations
            if stage in BAROSTAT_STAGES and solvation_mode != "implicit":
                payload["barostat"] = {
                    "type": "MonteCarloBarostat",
                    "pressure": (md_config.get("equilibration") or {})
                                .get("npt", {}).get("pressure", "1 bar"),
                }
            if stage == "REST2_1":
                payload["rest2"] = {
                    "tau_ladder": {
                        "minimum": production.tau_ladder.minimum,
                        "maximum": production.tau_ladder.maximum,
                        "count": production.tau_ladder.count,
                        "interpolation": production.tau_ladder.interpolation,
                        "tau_values": production.tau_ladder.tau_values(),
                    },
                    "derived_scale_factors": production.scale_factors(),
                    "exchange": {
                        "number_of_exchanges_per_segment":
                            production.exchange.number_of_exchanges_per_segment,
                        "duration_per_segment": production.duration_per_segment.source,
                        "exchange_interval_derived_ps": plan.steps_per_exchange * dt.value,
                        "steps_per_exchange": plan.steps_per_exchange,
                        "steps_per_exchange": plan.steps_per_exchange,
                    },
                    "enhanced_region": {"type": production.enhanced_region.type},
                    "omega_exclusion": {
                        "enabled": production.omega_exclusion.enabled,
                        "definition": production.omega_exclusion.definition,
                    },
                    "segment_note": (
                        "exchange_interval is DERIVED as duration_per_segment / "
                        "number_of_exchanges_per_segment, exactly in steps. How many segments to "
                        "run is set by NUMBER_OF_SEGMENTS "
                        "in run_all.sh, and completed segments are recorded in this run's manifest."
                    ),
                }
            _write_json(d / f"{stage}.json", payload)
            _write_script(d / f"{stage}.sh", _stage_launcher(stage, interpreter, pythonpath))
            if external_producer is not None:
                payload["input"]["state_sha256"] = inheritance["inherited_state_sha256"]
            stage_records.append({"stage": stage, "steps": stage_steps.get(stage, 0),
                                  "consumes": payload["input"]["state"],
                                  "produced_by": payload["input"]["produced_by"]})
            previous_state = f"{stage}/{stage}_final_state.xml"
            external_producer = None

        _write_script(staging / "run_all.sh", _run_all(stages_to_generate))
        (staging / "run.log").write_text(
            f"# generated {datetime.now(timezone.utc).isoformat()} by MD_input_gen.py\n"
            "# lines below are appended by the stage launchers as they execute\n")

        run_manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "MD_input_gen.py",
            "system": {
                "id": manifest["system"]["id"],
                "manifest_sha256": _sha256(system_manifest),
                "n_solute_atoms": manifest["composition"]["n_solute_atoms"],
            },
            "stages": stage_records,
            "resolved_md_config": resolution["resolved"],
            "configuration_hashes": resolution["hashes"],
            "value_sources": resolution["sources"],
            "generated_with": {"interpreter": interpreter,
                               "pythonpath": pythonpath or None},
            "randomness": seed_records,
            "generated_not_executed": True,
            "completed_segments": 0,
            "lineage": inheritance,
        }
        _write_json(staging / "run_manifest.json", run_manifest)

        publish(staging, outdir, overwrite=overwrite, what="project")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {"system_id": manifest["system"]["id"],
            "n_solute_atoms": manifest["composition"]["n_solute_atoms"],
            "stages": list(stages_to_generate), "summary": summary}
