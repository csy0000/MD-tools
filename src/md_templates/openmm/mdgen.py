"""Write a runnable MD project from a built system and `md.config.yaml`.

One directory per stage, in the order they depend on each other:

    MD/
      minimization/                                       the common chain
      eq/nvt_1kcal/  eq/npt_1kcal/  eq/npt_free/          equilibration, grouped
      cMD/                                                production, both branching from
      REST2/                                              the LAST common stage -- siblings

Each common stage reads its parent's `final_state.xml` and writes its own. `cMD` and `REST2` both
branch from the LAST common stage, so neither has to run before the other and REST2 never repeats
the minimisation or the NVT/NPT preparation.

The generated project contains ordinary OpenMM scripts and the configuration they read. It does not
import this package at run time and does not carry a copy of it: the classification work was done
by `sys-gen` and is in `inputs/solute.yaml`, and the stage helper plus the REST2 scaling arithmetic
travel as two small readable modules beside the scripts that use them.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from . import ais as A
from . import md_data_contract as MD
from .config import ConfigError, check_timestep_against_masses, resolve_md_config, \
    sha256_of_document, write_yaml
from .defaults import canonical_method
from .provenance_min import (implementation_identity, package_provenance, sha256_file,
                             template_identity)
from .stages import stage_plan

TEMPLATES = Path(__file__).resolve().parent / "templates"


def _template_module():
    """Load `templates/md_stages.py` from the same place md-gen copies it from.

    By path rather than by import, for two reasons. `templates/` is package DATA -- it has no
    `__init__.py` and nothing else in the package imports it -- so making it a namespace package
    here would be a new shape for the wheel to get right. And the point is that md-gen and the
    generated launcher share ONE derivation of the stage request: loading the same file they will
    copy is the most direct statement of that, and it cannot drift from what lands in the project.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_md_templates_stage_helpers", TEMPLATES / "md_stages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

#: What sys-gen writes and md-gen needs. A missing one is named rather than discovered later.
REQUIRED_INPUTS = ("system.xml", "topology.pdb", "solute.pdb", "initial_state.xml", "solute.yaml",
                   "resolved_sys.config.yaml")


def _executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _interpreter() -> str:
    """The python that generated this project, recorded so run.sh finds OpenMM by default.

    run.sh falls back to whatever `python3` resolves to if this path stops existing, so recording
    it is a convenience rather than a dependency -- a moved project still runs.
    """
    return sys.executable


def generate_md(*, input_folder: Path, config_path: Path, output_folder: Path) -> dict[str, Any]:
    inputs = Path(input_folder).resolve()
    out = Path(output_folder).resolve()
    config_path = Path(config_path).resolve()

    missing = [name for name in REQUIRED_INPUTS if not (inputs / name).is_file()]
    if missing:
        raise ConfigError(
            f"{inputs} is missing {', '.join(missing)}. Build the system first:\n"
            f"    md-openmm sys-gen -i <structure> --config sys.config.yaml -of {inputs}")

    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sys_resolved = yaml.safe_load((inputs / "resolved_sys.config.yaml").read_text())
    implicit = sys_resolved.get("solvation") == "implicit"
    # The dataset identity has ONE declaration, in sys.config.yaml, and `sys-gen` recorded the
    # resolved form beside the prepared system. Reading it back is what keeps md-gen from being a
    # second place a user has to state who owns this data.
    dataset_block = dict(sys_resolved.get("dataset") or {})

    resolved = resolve_md_config(document, implicit=implicit)
    check_timestep_against_masses(resolved, sys_resolved)
    methods = [canonical_method(m) for m in resolved["methods"]]

    out.mkdir(parents=True, exist_ok=True)
    # The inputs folder is addressed RELATIVE to the generated project, so moving `inputs/` and
    # `MD/` together needs no edit. An absolute path is recorded only if the two are not on a
    # shared root, in which case relativity would be a lie.
    try:
        relative_inputs = os.path.relpath(inputs, out)
    except ValueError:                             # different drives (Windows)
        relative_inputs = str(inputs)
    resolved["paths"] = {"inputs_folder": relative_inputs}

    # Contract check before any file is written, for the same reason sys-gen does it first.
    dataset_plan = _plan_dataset(dataset_block, out, methods)

    plan = stage_plan(resolved, implicit=implicit)
    resolved["paths"]["common_stages"] = [stage["path"] for stage in plan]
    resolved["paths"]["common_final_stage"] = plan[-1]["path"]
    # Both production methods read this one file. Written into the config rather than recomputed by
    # each script, so "where does production start" has exactly one answer in the project. cMD/ and
    # REST2/ sit one level under MD/, so `../` reaches the grouped stage directory.
    resolved["paths"]["common_final_state"] = f"../{plan[-1]['path']}/final_state.xml"
    # Recorded once, read by every generated script. REST2's per-tau equilibration used to look
    # for a key that was never written and recorded null.
    # Carried INTO the project so runtime records can name the implementation that wrote them.
    # The scripts cannot import md_templates to ask, and a run months later should not have to
    # guess which version produced it.
    # Whether this project belongs to a contract-managed dataset, recorded WITHOUT the absolute
    # value of MD_DATA: the generated tree must survive being moved and the variable changing.
    # `dataset.yaml` sits at the project root in contract mode, so nothing else is needed.
    resolved["dataset"] = {
        "contract_managed": bool(dataset_plan["contract_managed"]),
        "manifest": MD.MANIFEST_NAME if dataset_plan["contract_managed"] else None,
        "note": (None if dataset_plan["contract_managed"]
                 else "unregistered local project: not MD-data compliant"),
    }
    # ONE resolution of the generator identity, written identically everywhere. Reaching for
    # `implementation_identity()["git_commit"]` here used to produce null in a VCS-installed
    # package while the dataset manifest carried the verified commit from direct_url.json.
    resolved["provenance"] = _template_provenance()

    write_yaml(out / "md.config.yaml", resolved,
               header="# Resolved protocol, read by every run.py in this project.\n")

    # One copy for the whole project. Every script locates MD/ by looking for md.config.yaml above
    # itself, so stages at different depths all find the same helper.
    shutil.copy2(TEMPLATES / "md_stages.py", out / "md_stages.py")
    # The preflight every launcher runs before a Context exists. One copy for the whole project,
    # found the same way md_stages.py is.
    shutil.copy2(TEMPLATES / "preflight.py", out / "preflight.py")

    seed = _base_seed(resolved)
    for index, stage in enumerate(plan):
        directory = out / stage["path"]
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TEMPLATES / "stage_run.py", directory / "run.py")

        stage_document = dict(stage)
        stage_document["input_state"] = (
            stage["input_state"] if index else
            os.path.join(os.path.relpath(inputs, directory), "initial_state.xml"))
        stage_document.update({
            "temperature_kelvin": resolved["common"]["temperature_kelvin"],
            "timestep_fs": resolved["common"]["timestep_fs"],
            "friction_per_ps": resolved["common"]["friction_per_ps"],
            "integrator_seed": _seed(seed, stage["name"], "integrator"),
            "velocity_seed": _seed(seed, stage["name"], "velocities"),
            "barostat_seed": _seed(seed, stage["name"], "barostat"),
            # Only the first stage after minimisation assigns fresh velocities; every later stage
            # inherits them through its parent's final_state.xml.
            "assign_velocities": False,
            "template_commit": _template_commit(),
        })
        write_yaml(directory / "stage.yaml", stage_document,
                   header=f"# Stage {index + 1} of {len(plan)}. Read by run.py beside this file.\n")
        _write_launcher(TEMPLATES / "stage_run.sh", directory / "run.sh", stage["path"])

    for method in methods:
        directory = out / method
        directory.mkdir(parents=True, exist_ok=True)

        # Production stages get a stage.yaml, exactly as the common chain does. Without it the
        # generated launcher has nothing to hand the shared preflight, so `check_parent` and
        # `check_own_completion` short-circuit on `if not stage` and production runs with neither
        # -- which is how a missing parent came to be caught only after a Context existed, and a
        # changed request not at all.
        #
        # AIS gets one too, with a NULL parent: it starts from an equilibrium ensemble prepared
        # into inputs/, not from the common chain, and `check_parent` skips a stage that declares
        # none. `path_definition.yaml` still records the resolved path for the run itself; this is
        # the shared contract every production method now states the same way, so the preflight and
        # the consumer gate have one shape to check rather than one per method.
        if True:
            parent = None if method == "AIS" else resolved["paths"]["common_final_stage"]
            stage_document = _template_module().production_stage_document(
                resolved, method,
                parent_stage=parent,
                parent_path=None if parent is None else os.path.join("..", parent),
                implicit=implicit,
                seeds={"integrator": _seed(seed, method, "integrator"),
                       "velocities": _seed(seed, method, "velocities"),
                       "barostat": _seed(seed, method, "barostat")},
                template_commit=_template_commit(),
            )
            write_yaml(directory / "stage.yaml", stage_document,
                       header="# The resolved PRODUCTION request. Read by run.py beside this\n"
                              "# file and by the shared preflight. Its invariant fingerprint is\n"
                              "# what a continuation is checked against: running longer is a\n"
                              "# legitimate extension, changing the physics is not.\n")

        if method == "cMD":
            shutil.copy2(TEMPLATES / "cmd_run.py", directory / "run.py")
            # cMD carries the scaling module because it may run at tau > 0. It is the SAME file
            # REST2 gets, so a fixed-tau walker cannot drift from the ladder it is meant to match.
            shutil.copy2(TEMPLATES / "rest2_scaling.py", directory / "rest2_scaling.py")
            _write_launcher(TEMPLATES / "stage_run.sh", directory / "run.sh", "cMD production")
        elif method == "AIS":
            # AIS starts from an existing equilibrium trajectory, not from the common chain, so it
            # gets the same scaling module and nothing else. Only the four files below exist after
            # md-gen; the trajectory directories and every runtime record are written by the run.
            shutil.copy2(TEMPLATES / "ais_run.py", directory / "run.py")
            shutil.copy2(TEMPLATES / "rest2_scaling.py", directory / "rest2_scaling.py")
            # The shared source-ensemble reader. AIS and rREST2 both draw configurations out of an
            # equilibrium production run, and the rules for doing that safely live in one file so
            # a second implementation cannot drift from it.
            shutil.copy2(TEMPLATES / "source_ensemble.py", directory / "source_ensemble.py")
            write_yaml(directory / "path_definition.yaml",
                       _ais_path_definition(resolved, implicit=implicit),
                       header="# The AIS path, its schedule and its work convention, resolved\n"
                              "# once by md-gen. run.py beside this file reads it; it does not\n"
                              "# recompute it.\n")
            _write_launcher(TEMPLATES / "stage_run.sh", directory / "run.sh",
                            "AIS switching paths")
        else:
            # The LEGACY exchange loop. The production REST2 engine is OpenMMTools, reached
            # through `md-openmm setup` with `protocol: REST2`; this route keeps its own loop
            # because the datasets already generated through it depend on its on-disk layout,
            # and invalidating their provenance would be worse than keeping a legacy path that
            # says so. The two agree by construction: one `exchange_log_acceptance`, one
            # `exchange_pairs`, and a test that their decisions match. See
            # docs/openmmtools-rest2.md.
            shutil.copy2(TEMPLATES / "rest2_run.py", directory / "run.py")
            shutil.copy2(TEMPLATES / "rest2_equilibrate.py", directory / "equilibrate.py")
            shutil.copy2(TEMPLATES / "rest2_scaling.py", directory / "rest2_scaling.py")
            _write_launcher(TEMPLATES / "stage_run.sh", directory / "run.sh",
                            "REST2 exchange production")
            _write_launcher(TEMPLATES / "stage_run.sh", directory / "equilibrate.sh",
                            "REST2 per-tau equilibration", script="equilibrate.py")
            shutil.copy2(TEMPLATES / "extend.sh", directory / "extend.sh")
            _executable(directory / "extend.sh")
            for replica in range(int(resolved["REST2"]["number_of_replicas"])):
                for phase in ("equilibration", "production"):
                    (directory / f"replica_{replica:02d}" / phase).mkdir(parents=True,
                                                                        exist_ok=True)

    _write_run_all(out, plan, methods)
    dataset = _write_dataset_manifest(dataset_plan, echo=True)

    write_yaml(out / "provenance.yaml", _md_provenance(
        dataset=dataset,
        out=out, inputs=inputs, relative_inputs=relative_inputs, resolved=resolved,
        sys_resolved=sys_resolved, plan=plan, methods=methods, seed=seed))
    manifest = write_generated_manifest(out)
    if manifest is not None:
        pass
    return {"output_folder": str(out), "methods": methods, "implicit": implicit,
            "common_stages": [stage["path"] for stage in plan], "dataset": dataset}


MD_PROVENANCE_FORMAT = "md-templates-md-provenance/v1"
#: What `md-gen` itself wrote. NOT the dataset checksum manifest: trajectories, checkpoints and
#: final states do not exist yet when this is written, and MD-data computes those at archival.
GENERATED_MANIFEST = "generated-files.sha256"


def implicit_route(sys_resolved: dict) -> bool:
    return sys_resolved.get("solvation") == "implicit"


def _md_provenance(*, out: Path, inputs: Path, relative_inputs: str, resolved: dict,
                   sys_resolved: dict, plan: list, methods: list, seed: int,
                   dataset: Optional[dict] = None) -> dict[str, Any]:
    """Which implementation generated this project, from which prepared system, with which seeds.

    The lineage fields -- the three hashes of the parent `inputs/` records -- are what let a reader
    confirm that this MD/ belongs to that inputs/, rather than to a different bundle that happens
    to sit beside it.
    """
    from .provenance_min import (environment_versions, implementation_identity, sha256_file,
                                 template_identity)

    def parent_hash(name: str) -> Optional[str]:
        path = inputs / name
        return sha256_file(path) if path.is_file() else None

    stage_seeds = {}
    for stage in plan:
        stage_seeds[stage["path"]] = {
            "integrator": _seed(seed, stage["name"], "integrator"),
            "velocities": _seed(seed, stage["name"], "velocities"),
            "barostat": _seed(seed, stage["name"], "barostat"),
        }
    replica_seeds = {}
    if "REST2" in methods:
        for replica in range(int((resolved.get("REST2") or {}).get("number_of_replicas", 0))):
            replica_seeds[f"replica_{replica:02d}"] = {
                name: _seed(seed, "REST2", replica, name)
                for name in ("integrator", "velocities", "barostat")}

    common = resolved.get("common") or {}
    constraints = (sys_resolved.get("constraints") or {})
    timestep_fs = float(common.get("timestep_fs", 2.0))
    frequency = common.get("barostat_frequency_steps")
    return {
        "format": MD_PROVENANCE_FORMAT,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": list(sys.argv),
        "implementation": implementation_identity(),
        # The canonical generator identity, resolved once. `implementation` is the raw
        # observation; this is the answer every record in the project agrees on, and it
        # carries the direct_url.json commit that `implementation.git_commit` is null for
        # in a VCS-installed package.
        "template": template_identity(),
        "environment": environment_versions(),
        # Whether this project belongs to a contract-managed MD-data dataset, and which validator
        # said so. An unregistered local tree says so rather than leaving it to be assumed.
        "dataset": dataset,
        # The protocol as RESOLVED, recorded here rather than left to be read back out of the
        # configuration. `null` means "does not apply to this solvation route", which is why the
        # implicit case writes null for pressure and barostat rather than omitting them.
        "protocol": {
            "thermostat": {
                "integrator": "openmm.LangevinMiddleIntegrator",
                "temperature_kelvin": common.get("temperature_kelvin"),
                "friction_per_ps": common.get("friction_per_ps"),
                "friction_note": ("OpenMM collision rate in ps^-1; 1.0 ps^-1 is a nominal 1 ps "
                                  "damping time"),
            },
            "timestep_fs": timestep_fs,
            "constraints": {
                "type": constraints.get("type"),
                "rigid_water": constraints.get("rigid_water"),
                "hydrogen_mass_amu": constraints.get("hydrogen_mass_amu"),
                "hydrogen_mass_repartitioning": constraints.get("hydrogen_mass_amu") is not None,
            },
            "pressure_coupling": None if implicit_route(sys_resolved) else {
                "barostat": "openmm.MonteCarloBarostat",
                "pressure_bar": common.get("pressure_bar"),
                "frequency_steps": frequency,
                "interval_ps": (round(float(frequency) * timestep_fs / 1000.0, 6)
                                if frequency else None),
                "active_in_stages": [s["path"] for s in plan if s.get("barostat_active")],
                "present_but_inactive_in_stages": [s["path"] for s in plan
                                                   if not s.get("barostat_active")],
            },
        },
        "parent_system": {
            "inputs_path": relative_inputs,
            "provenance_sha256": parent_hash("provenance.yaml"),
            "forcefield_sha256": parent_hash("forcefield.json"),
            "checksums_sha256": parent_hash("SHA256SUMS"),
            "input_hashes": {name: sha256_file(inputs / name) for name in REQUIRED_INPUTS},
        },
        "sys_config_hash": sha256_of_document(sys_resolved),
        # The RESOLVED protocol this project runs, hashed from the bytes written to
        # MD/md.config.yaml -- resolution fills in stage paths and reconciles the ensemble with
        # the solvent, so the input document is a different file.
        "md_config_hash": sha256_file(out / "md.config.yaml"),
        "methods": list(methods),
        "common_stage_plan": [{"path": s["path"], "kind": s["kind"], "ensemble": s["ensemble"],
                               "parent": s.get("parent_path"),
                               "input_state": s["input_state"]} for s in plan],
        "seeds": {"base": seed, "common_stages": stage_seeds, "rest2_replicas": replica_seeds},
        "generated_paths": sorted(
            p.relative_to(out).as_posix() for p in out.rglob("*")
            if p.is_file() and p.name != GENERATED_MANIFEST),
        "checksum_manifest": GENERATED_MANIFEST,
        "checksum_manifest_scope": (
            "files written by md-gen before any dynamics: scripts, launchers, stage.yaml and "
            "md.config.yaml. Trajectories, checkpoints and final states are produced later and "
            "are checksummed by MD-data at archival, not here."),
    }


def write_generated_manifest(out: Path) -> Path:
    """Hash what md-gen produced, deterministically, excluding the manifest itself."""
    from .provenance_min import sha256_file

    out = Path(out)
    lines = []
    for path in sorted(out.rglob("*"), key=lambda p: p.relative_to(out).as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(out).as_posix()
        if relative == GENERATED_MANIFEST:
            continue
        lines.append(f"{sha256_file(path)}  {relative}")
    manifest = out / GENERATED_MANIFEST
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _template_commit() -> str | None:
    """The exact commit of the MD-templates that wrote this project, or None.

    From the canonical resolution, so a VCS install records the commit `direct_url.json` proves
    rather than the null a checkout-only lookup returns.
    """
    return template_identity()["commit"]


def _template_provenance() -> dict[str, Any]:
    """The generator identity as it is written into every record that names it."""
    identity = template_identity()
    return {
        "template_commit": identity["commit"],
        "template_commit_evidence": identity["evidence"],
        "template_repository": identity["repository"],
        "md_templates_version": identity["version"],
        "installed_fingerprint": identity["installed_fingerprint"],
        "git_dirty": identity["dirty"],
        # Spelled out where an unregistered project records it, because a commit beside a dirty
        # tree looks exactly like a commit that reproduces the run.
        "reproducible_from_commit": (None if identity["commit"] is None
                                     else not identity["dirty"]),
    }


def _base_seed(resolved: dict[str, Any]) -> int:
    value = (resolved.get("common") or {}).get("random_seed")
    return int(value) if value is not None else 20260825


def _seed(base: int, *purpose: Any) -> int:
    """The same derivation the generated scripts use, so a stage.yaml and a run agree."""
    value = int(base)
    for part in purpose:
        for byte in str(part).encode("utf-8"):
            value = (value * 1000003 + byte) & 0xFFFFFFFF
    seed = value % (2 ** 31 - 1)
    return seed or 1


def _write_launcher(template: Path, path: Path, label: str, *, script: str = "run.py") -> None:
    text = (template.read_text()
            .replace("__PYTHON__", _interpreter())
            .replace("__STAGE__", label)
            .replace("__SCRIPT__", script)
            .replace("__LOG__", Path(script).stem + ".log"))
    path.write_text(text, encoding="utf-8")
    _executable(path)


#: Which MD-data component each generated directory belongs to. The equilibration STAGES are not
#: components: `eq/nvt_1kcal` lives inside the `eq` component, and declaring one component per
#: stage would turn one equilibration into four datasets.
COMPONENT_OF_METHOD = {"cMD": "cMD", "REST2": "REST2", "AIS": "AIS"}
COMMON_COMPONENTS = ("minimization", "eq")


def _plan_dataset(dataset_block: dict[str, Any], out: Path,
                  methods: list[str]) -> dict[str, Any]:
    """The components this generation adds, validated against the dataset that already exists.

    `sys-gen` created the dataset and declared `common`. This adds `minimization`, `eq`, and one
    component per selected production method -- to the SAME manifest, not a second one.
    """
    if not dataset_block.get("enabled"):
        return {"contract_managed": False,
                "note": ("unregistered local generation: no dataset.yaml, and NOT MD-data "
                         "compliant."),
                "validator": MD.md_data_identity()}

    MD.require_md_data()
    # `out` IS the dataset root for md-gen, not a component inside it.
    location = MD.resolve_roots(out, component=None)
    manifest_path = Path(location["dataset_root"]) / MD.MANIFEST_NAME
    if not manifest_path.is_file():
        raise ConfigError(
            f"dataset.enabled is true but {manifest_path} does not exist.\n"
            f"  `md-openmm sys-gen -of \"$MD_DATA_LOCAL/{MD.COMMON_COMPONENT}/\"` creates the "
            f"dataset and its manifest; md-gen adds method components to it.")
    existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    # md-gen does not re-ask for the identity, so the commit it checks is the one sys-gen already
    # recorded in the manifest. It must still be the generator that is running NOW: generating a
    # dataset's system with one checkout and its scripts with another produces a tree whose single
    # recorded provenance is true of only half of it.
    established = MD.check_templates_commit((existing.get("templates") or {}).get("commit"))

    components = [
        MD.component_entry("minimization", kind="simulation", method="minimization",
                           description="Restrained energy minimisation of the prepared system."),
        MD.component_entry("eq", kind="simulation", method="equilibration",
                           description="The equilibration chain. Its stages are directories "
                                       "inside this component, not components of their own."),
    ]
    for method in methods:
        components.append(MD.component_entry(
            COMPONENT_OF_METHOD[method], kind="simulation", method=method,
            description=f"{method} production."))

    manifest = dict(existing)
    manifest["components"] = MD.merge_components(existing.get("components") or [], components)
    MD.validate(manifest)                       # metadata only: the directories do not exist yet
    return {"contract_managed": True, "manifest": manifest, "location": location,
            "generator": established,
            "added": [entry["name"] for entry in components]}


def _write_dataset_manifest(plan: dict[str, Any], *, echo: bool = True) -> dict[str, Any]:
    """Update `dataset.yaml` with the new components, and validate the layout now it exists."""
    if not plan["contract_managed"]:
        return {"contract_managed": False, "note": plan["note"], "validator": plan["validator"]}

    location = plan["location"]
    path = MD.write_manifest(Path(location["dataset_root"]), plan["manifest"])
    report = MD.validate(plan["manifest"], root=location["md_data"],
                         dataset_root=location["dataset_root"])
    if echo:
        print(f"  dataset      : {report['dataset_id']} at {report['path']}, "
              f"components {', '.join(report['components'])}")
        print(f"  manifest     : {path.name} validated by md-data "
              f"{report['validator']['version']} (contract v"
              f"{report['validator']['contract_version']}), layout verified")
    # `md_data` / `md_data_local` are deliberately absent: they are this machine's storage
    # location, and the record must survive the tree being moved.
    return {"contract_managed": True, "manifest": MD.MANIFEST_NAME, **report}


def _ais_path_definition(resolved: dict[str, Any], *, implicit: bool) -> dict[str, Any]:
    """The full AIS path, resolved once here so the generated runtime never recomputes it.

    The schedule is derived arithmetic -- which taus the path visits, which of them are observed --
    and deriving it in two places is how a project ends up observing a schedule its own record does
    not describe. `md-gen` computes it; `AIS/run.py` reads it.
    """
    block = resolved["AIS"]
    path = dict(block["path"])
    common = resolved["common"]
    schedule = A.switching_schedule(
        tau_start=float(path["tau_start"]), tau_end=float(path["tau_end"]),
        switching_duration_ps=float(path["switching_duration_ps"]),
        parameter_update_interval_steps=int(path["parameter_update_interval_steps"]),
        number_of_observations=int(block["output"]["number_of_observations"]),
        timestep_fs=float(common["timestep_fs"]))
    return {
        "format": "md-templates-ais-path/v1",
        "path": path,
        "scaling": {
            # tau is the SOURCE parameter; the other two are labelled derived and are never read
            # back in as input.
            "source_parameter": "tau",
            "rest2_implementation": "rest2-no-bond-angle-omega/v1; tau is the only coordinate",
            "implementation": "rest2_scaling.TauSwitcher, the same decomposition a static REST2 "
                              "rung is built with",
            "enhanced_region": path["enhanced_region"],
            "omega_exclusion": bool(path["omega_exclusion"]),
            "omega_source": "inputs/solute.yaml rest2.omega_excluded_bonds",
        },
        "schedule": schedule,
        "output": dict(block["output"]),
        "execution": dict(block["execution"]),
        "work_convention": A.WORK_CONVENTION,
        "observation_convention": (
            "observation 0 is the source configuration at tau_start with zero cumulative work; "
            "observations 1..N-1 are the coordinates after propagating at their scheduled tau, "
            "paired with the cumulative work through the parameter change that reached it. One "
            "DCD frame per row, in the same order."),
        "ensemble": {
            "implicit_solvent": bool(implicit),
            "constant_volume": True,
            "barostat": None,
            "temperature_kelvin": common["temperature_kelvin"],
            "note": ("switching is at fixed volume: no barostat is added and an explicit source "
                     "frame keeps its own box. An NPT source ensemble may seed these paths, but "
                     "pressure-volume work is not part of this implementation."),
        },
        **_template_provenance(),
    }


def _write_run_all(out: Path, plan: list[dict[str, Any]], methods: list[str]) -> None:
    """The convenience wrapper. It calls the stage scripts; it does not reimplement them.

    AIS is deliberately absent. It does not start from the common equilibration chain -- it starts
    from an equilibrium trajectory the user already produced -- so running it "in order" with the
    rest would be running it before its own input exists.
    """
    lines, checks = [], []
    if "cMD" in methods:
        lines += ['echo "== cMD production =="', '( cd cMD && ./run.sh )']
        checks += ['    ( cd cMD && ./run.sh --check )']
    if "REST2" in methods:
        lines += ['echo "== REST2 per-tau equilibration =="',
                  '( cd REST2 && ./equilibrate.sh )',
                  'echo "== REST2 exchange production =="',
                  '( cd REST2 && ./run.sh )']
        checks += ['    ( cd REST2 && ./equilibrate.sh --check )',
                   '    ( cd REST2 && ./run.sh --check )']
    # AIS is absent from both: it starts from an equilibrium trajectory the user already produced,
    # so running it "in order" would run it before its own input exists. `MD/AIS/run.sh --check`
    # is how its preflight is run on its own.
    text = ((TEMPLATES / "run_all.sh").read_text()
            .replace("__COMMON_STAGES__", " ".join(stage["path"] for stage in plan))
            .replace("__PRODUCTION_CHECK__", "\n".join(checks) or "    :")
            .replace("__PRODUCTION__", "\n".join(lines)))
    (out / "run_all.sh").write_text(text, encoding="utf-8")
    _executable(out / "run_all.sh")
