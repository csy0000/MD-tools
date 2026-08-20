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
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = ["generate_project", "RUN_MANIFEST_SCHEMA_VERSION", "STAGE_ORDER"]

RUN_MANIFEST_SCHEMA_VERSION = 1

#: Stage directory names. Conventional MD is `cMD_N` and replica exchange is `REST2_N`, capitalised
#: as the methods are normally written. The equilibration stages are singular because the worked
#: protocol has one of each; the numbering exists on production stages because a protocol may have
#: several blocks.
STAGE_ORDER = ("min", "eq_nvt", "eq_npt", "cMD_1", "REST2_1")

#: Keys md_config.json may carry that the canonical model does not model. See _resolved_spec.
GENERATOR_ONLY_KEYS = ("conventional_md", "minimization")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _stage_launcher(stage: str, project: Path) -> str:
    """A readable launcher. It calls a package module; it does not reimplement any physics."""
    return f"""#!/usr/bin/env bash
# Stage: {stage}
#
# This script is the record of how this stage is run. It calls the package's stage command; the
# physics lives there, not here, so there is exactly one implementation to audit.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"

: "${{PYTHON:=python}}"
: "${{MD_DEVICES:=}}"

cd "$HERE"
echo "[{stage}] start $(date -Is)" | tee -a "$PROJECT/run.log"

"$PYTHON" -m md_templates.openmm.stage --config "{stage}.json" ${{MD_DEVICES:+--devices "$MD_DEVICES"}}

echo "[{stage}] done  $(date -Is)" | tee -a "$PROJECT/run.log"
"""


def _run_all(stages: tuple[str, ...]) -> str:
    lines = "\n".join(
        f'run_stage {s}' for s in stages
    )
    return f"""#!/usr/bin/env bash
# Run every stage in order. Each stage's own launcher owns how that stage runs; this file owns
# only the ORDER and the record of what was executed.
set -euo pipefail
PROJECT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$PROJECT"

# How many REST2 segments to run. This is an EXECUTION choice and lives here, never in the
# scientific JSON: asking for a longer run must not change the configuration hash.
: "${{NUMBER_OF_SEGMENTS:=2}}"
export NUMBER_OF_SEGMENTS

run_stage () {{
    local s="$1"
    if [ ! -x "$s/$s.sh" ]; then
        echo "missing launcher: $s/$s.sh" >&2; exit 1
    fi
    echo "=== $s ===" | tee -a run.log
    "./$s/$s.sh"
}}

{lines}

echo "all stages complete $(date -Is)" | tee -a run.log
"""


def generate_project(*, system_manifest: Path, md_config: dict, outdir: Path,
                     inherit: Optional[Path] = None, overwrite: bool = False,
                     dry_run: bool = False) -> dict:
    """Generate the staged project. Transactional; writes nothing on failure."""
    from .segments import plan_segment_from_exchanges, reporting_interval_steps, steps_for_duration

    manifest = _verify_bundle(system_manifest)
    bundle_dir = system_manifest.parent
    manifest["_bundle_dir"] = str(bundle_dir)

    resolution, forcefield = _resolved_spec(md_config, manifest)
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
    if equilibration.npt is not None:
        stage_steps["eq_npt"] = steps_for_duration(
            equilibration.npt.value, dt.value, duration_source=equilibration.npt.source,
            timestep_source=dt.source, duration_label="equilibration.npt")

    if production.method == "rest2":
        plan = plan_segment_from_exchanges(
            production.exchange.n_exchange_per_segment,
            production.exchange.exchange_interval.value, dt.value,
            interval_source=production.exchange.exchange_interval.source,
            timestep_source=dt.source)
        stage_steps["REST2_1"] = plan.steps_per_segment
        summary.append(
            f"REST2_1  {production.n_replicas} replicas, "
            f"{production.exchange.n_exchange_per_segment} x "
            f"{production.exchange.exchange_interval.source} = "
            f"{plan.steps_per_segment:,} steps per segment "
            f"({plan.steps_per_exchange:,} per exchange round)")
    else:
        raise ValueError(
            "md_config.json declares protocol.production.method "
            f"{production.method!r}; the staged layout generates a REST2 production stage. "
            "Conventional-MD-only projects are not yet generated."
        )

    cmd_ns = (md_config.get("conventional_md") or {}).get("duration", "1 ns")
    from .spec.units import parse_quantity
    cmd_duration = parse_quantity(cmd_ns, dimension="time")
    stage_steps["cMD_1"] = steps_for_duration(
        cmd_duration.value, dt.value, duration_source=cmd_duration.source,
        timestep_source=dt.source, duration_label="conventional_md.duration")
    stage_steps["min"] = 0

    reporting = spec.execution.reporting
    full_steps = reporting_interval_steps(
        reporting.all_atom.value, dt.value, interval_source=reporting.all_atom.source,
        timestep_source=dt.source, label="reporting.all_atom")
    selected_steps = reporting_interval_steps(
        reporting.solute.value, dt.value, interval_source=reporting.solute.source,
        timestep_source=dt.source, label="reporting.solute")

    summary.append(f"min      max {equilibration.minimize_max_iterations} iterations")
    summary.append(f"eq_nvt   {stage_steps.get('eq_nvt', 0):,} steps ({equilibration.nvt.source})"
                   if equilibration.nvt else "eq_nvt   (not configured)")
    summary.append(f"eq_npt   {stage_steps.get('eq_npt', 0):,} steps ({equilibration.npt.source})"
                   if equilibration.npt else "eq_npt   (not configured)")
    summary.append(f"cMD_1    {stage_steps['cMD_1']:,} steps ({cmd_duration.source})")
    summary.append(f"reporting: full system every {full_steps:,} steps, "
                   f"selected atoms every {selected_steps:,} steps")

    if dry_run:
        return {"system_id": manifest["system"]["id"],
                "n_solute_atoms": manifest["composition"]["n_solute_atoms"],
                "stages": list(STAGE_ORDER), "summary": summary}

    # ---- transactional generation ---------------------------------------------------------------
    staging = outdir.parent / f".{outdir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        inputs = staging / "inputs"
        inputs.mkdir()
        for name in ("system.xml", "topology.pdb", "topology.cif", "initial_state.xml",
                     "forcefield.json", "system_manifest.json", "checksums.json"):
            shutil.copy2(bundle_dir / name, inputs / name)

        restraint = (md_config.get("minimization") or {}).get("restraint") or {}
        restraint_k = restraint.get("force_constant_kcal_per_mol_angstrom2", 1.0)

        previous_state = "inputs/initial_state.xml"
        stage_records = []
        for stage in STAGE_ORDER:
            d = staging / stage
            d.mkdir()
            restrained = stage in ("min", "eq_nvt", "eq_npt")
            payload = {
                "stage": stage,
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                # every stage NAMES what it consumes and which stage produced it
                "input": {
                    "system_xml": "../inputs/system.xml",
                    "topology": "../inputs/topology.pdb",
                    "state": f"../{previous_state}" if previous_state.startswith("inputs/")
                             else f"../{previous_state}",
                    "produced_by": ("MD_system_gen.py" if previous_state.startswith("inputs/")
                                    else previous_state.split("/")[0]),
                },
                "output": {
                    "final_state": f"{stage}_final_state.xml",
                    "final_structure": f"{stage}_final.pdb",
                    "checkpoint": f"{stage}.chk",
                    "log": f"{stage}.log",
                    "results": f"{stage}_results.json",
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
                    "full_system_interval_steps": full_steps,
                    "selected_atoms_interval_steps": selected_steps,
                    "selected_atoms": {"type": "solute"},
                },
                "execution": {
                    "platform": spec.execution.platform,
                    "precision": spec.execution.precision,
                },
            }
            if stage == "min":
                payload["max_iterations"] = equilibration.minimize_max_iterations
            if stage == "eq_npt":
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
                        "n_exchange_per_segment":
                            production.exchange.n_exchange_per_segment,
                        "exchange_interval":
                            production.exchange.exchange_interval.source,
                        "steps_per_exchange": plan.steps_per_exchange,
                    },
                    "enhanced_region": {"type": production.enhanced_region.type},
                    "omega_exclusion": {
                        "enabled": production.omega_exclusion.enabled,
                        "definition": production.omega_exclusion.definition,
                    },
                    "segment_note": (
                        "duration_per_segment is DERIVED as n_exchange_per_segment x "
                        "exchange_interval. How many segments to run is set by NUMBER_OF_SEGMENTS "
                        "in run_all.sh, and completed segments are recorded in this run's manifest."
                    ),
                }
            _write_json(d / f"{stage}.json", payload)
            _write_script(d / f"{stage}.sh", _stage_launcher(stage, staging))
            stage_records.append({"stage": stage, "steps": stage_steps.get(stage, 0),
                                  "consumes": payload["input"]["state"]})
            previous_state = f"{stage}/{stage}_final_state.xml"

        _write_script(staging / "run_all.sh", _run_all(STAGE_ORDER))
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
            "generated_not_executed": True,
            "completed_segments": 0,
            "lineage": ({"inherited_from": str(inherit),
                         "inherited_run_manifest_sha256": _sha256(inherit),
                         "note": "lineage only. This is NOT a checkpoint resume: continuing a "
                                 "REST2 run happens in that run's own directory through the "
                                 "committed-generation contract."}
                        if inherit else None),
        }
        _write_json(staging / "run_manifest.json", run_manifest)

        if outdir.exists():
            if not overwrite:
                raise RuntimeError(f"{outdir} appeared during generation; refusing to overwrite")
            shutil.rmtree(outdir)
        staging.replace(outdir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {"system_id": manifest["system"]["id"],
            "n_solute_atoms": manifest["composition"]["n_solute_atoms"],
            "stages": list(STAGE_ORDER), "summary": summary}
