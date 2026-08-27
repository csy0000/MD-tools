"""Everything checked before a single integration step. Copied into every generated project.

    ./run.sh --check          this, and nothing else
    ./run.sh                  this, automatically, and then the stage

A run that is going to fail should fail in a second, before an OpenMM `Context` exists -- not after
minimisation, not after the first worker process, and above all not after a trajectory file has
been created. Every generated launcher calls `require()` at the top of `main()`, before the System
is deserialized and before any Context, integrator, worker, checkpoint or reporter is constructed.

**Bounded and non-destructive, deliberately.** This reads the named manifest, the declared component
directories, the small configuration and provenance files beside it, and specifically named source
and output artifacts. It never walks `$MD_DATA`, never opens another dataset, never hashes a
production trajectory, and never writes anything.

MD-data owns the dataset contract. When this project is contract-managed the manifest is validated
by `md_data`'s own validator, imported at runtime; there is no copy of its schema here. When the
project is an unregistered local `inputs/ + MD/` tree the dataset checks are skipped and say so --
such a tree is not MD-data compliant and is not treated as though it were.

This module imports OpenMM, PyYAML and optionally `md_data`. Never `md_templates`.
"""
import json
import os
import sys
from pathlib import Path

import yaml

PASS, FAIL, SKIP = "PASS", "FAIL", "skip"

#: Which declared component a generated directory belongs to. The equilibration STAGES are not
#: components: `eq/nvt_1kcal` lives inside the `eq` component, and one component per stage would
#: turn one equilibration into four datasets.
COMPONENT_OF_DIRECTORY = {
    "minimization": "minimization",
    "eq": "eq",
    "cMD": "cMD",
    "REST2": "REST2",
    "AIS": "AIS",
}

#: What `sys-gen` writes and every generated script needs. A missing one is named here rather than
#: discovered halfway through a build.
REQUIRED_COMMON = ("system.xml", "topology.pdb", "solute.pdb", "initial_state.xml", "solute.yaml",
                   "resolved_sys.config.yaml")


class Result:
    """One row of the check table."""

    def __init__(self, name, status, detail=""):
        self.name, self.status, self.detail = name, status, detail

    @property
    def failed(self):
        return self.status == FAIL

    def __repr__(self):                                       # pragma: no cover
        return f"<{self.name}: {self.status}>"


def _ok(name, detail=""):
    return Result(name, PASS, detail)


def _no(name, detail):
    return Result(name, FAIL, detail)


def _skip(name, detail):
    return Result(name, SKIP, detail)


# ---------------------------------------------------------------------------------------------
# The dataset contract
# ---------------------------------------------------------------------------------------------

def _component_for(here, project):
    """Which declared component this directory belongs to, from its path under the project."""
    try:
        relative = here.resolve().relative_to(project.resolve())
    except ValueError:
        return None
    parts = relative.parts
    return COMPONENT_OF_DIRECTORY.get(parts[0]) if parts else None


def check_dataset(here, project, config):
    """The MD-data half: roots, manifest, status, and this component's declaration.

    Skipped entirely, and said to be skipped, when the project is not contract-managed.
    """
    block = dict((config.get("dataset") or {}))
    if not block.get("contract_managed"):
        return [_skip("dataset contract",
                      "unregistered local project: no dataset.yaml, not MD-data compliant")]

    results = []
    root = os.environ.get("MD_DATA")
    local = os.environ.get("MD_DATA_LOCAL")
    if not root:
        return [_no("MD_DATA", "not set; it is the managed-storage root every canonical dataset "
                               "path is relative to")]
    if not local:
        return [_no("MD_DATA_LOCAL", "not set; it is the one dataset root this run belongs to")]
    root_path, local_path = Path(root).expanduser(), Path(local).expanduser()
    if not root_path.is_dir():
        return [_no("MD_DATA", f"{root!r} is not an existing directory")]
    root_path, local_path = root_path.resolve(), local_path.resolve()
    try:
        relative = local_path.relative_to(root_path)
    except ValueError:
        return [_no("MD_DATA_LOCAL", f"{local!r} is not inside MD_DATA {root!r}")]
    results.append(_ok("MD_DATA / MD_DATA_LOCAL", f"{relative.as_posix()} under {root_path}"))

    # The dataset root this run is actually in must BE the selected one. Comparing the resolved
    # project directory against MD_DATA_LOCAL is what stops a run in one dataset from validating
    # another dataset's manifest and believing itself checked.
    if project.resolve() != local_path:
        return results + [_no(
            "dataset root",
            f"this project is at {project.resolve()} but MD_DATA_LOCAL is {local_path}")]
    results.append(_ok("dataset root", "the running project is the selected dataset"))

    manifest = project / "dataset.yaml"
    if not manifest.is_file():
        return results + [_no("dataset.yaml", f"{manifest} does not exist")]

    try:
        import md_data
    except ImportError as error:
        return results + [_no(
            "md-data validator",
            f"the md-data package is not installed ({type(error).__name__}), so this dataset "
            f"cannot be validated against the contract that owns it")]

    try:
        document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        dataset = md_data.validate_dataset(document)
    except md_data.ContractViolation as error:
        return results + [_no("md-data validator",
                              f"{len(error.problems)} contract violation(s): "
                              + "; ".join(error.problems[:3]))]
    except Exception as error:
        return results + [_no("md-data validator", f"{type(error).__name__}: {error}")]

    from md_data.storage import check_dataset_tree

    try:
        problems = check_dataset_tree(root_path, dataset)
    except md_data.ContractViolation as error:
        problems = list(error.problems)
    if problems:
        return results + [_no("declared layout", "; ".join(problems[:3]))]
    results.append(_ok("md-data validator",
                       f"{dataset.dataset_id} valid, contract v{md_data.CONTRACT_VERSION}, "
                       f"layout verified"))

    if dataset.status != "active":
        return results + [_no("dataset status",
                              f"{dataset.status!r}: a complete or archived dataset is read-only, "
                              f"and new output belongs in a new dated dataset")]
    if dataset.role == "baseline" and dataset.namespace != "baseline":
        return results + [_no("role/namespace", "role baseline outside the baseline namespace")]
    if dataset.role != "baseline" and dataset.namespace == "baseline":
        return results + [_no("role/namespace",
                              "a project dataset in the reserved baseline namespace")]
    results.append(_ok("dataset status", f"active, role {dataset.role}, writable"))

    wanted = _component_for(here, project)
    if wanted is None:
        return results + [_no("component", f"{here} is not inside a known component of {project}")]
    declared = {c.name: c for c in dataset.components}
    component = declared.get(wanted)
    if component is None:
        return results + [_no("component",
                              f"{wanted!r} is not declared in dataset.yaml (declared: "
                              f"{', '.join(sorted(declared)) or 'none'})")]
    if component.linked:
        return results + [_no("component",
                              f"{wanted!r} is linked: true -- a linked component is content "
                              f"another dataset owns and is never writable")]
    if component.status not in ("active",):
        return results + [_no("component",
                              f"{wanted!r} has status {component.status!r}; new output belongs in "
                              f"an active component this dataset owns")]
    results.append(_ok("component", f"{wanted!r} declared, owned, active, not linked"))
    return results


# ---------------------------------------------------------------------------------------------
# The prepared system
# ---------------------------------------------------------------------------------------------

def check_common(inputs, config):
    """`common/` holds what the scripts need, and its own checksum record still agrees."""
    results = []
    missing = [name for name in REQUIRED_COMMON if not (inputs / name).is_file()]
    if missing:
        return [_no("prepared inputs",
                    f"{inputs} is missing {', '.join(missing)} -- build the system with "
                    f"`md-openmm sys-gen`")]
    results.append(_ok("prepared inputs", f"{len(REQUIRED_COMMON)} required file(s) in "
                                          f"{inputs.name}/"))

    manifest = inputs / "SHA256SUMS"
    if not manifest.is_file():
        results.append(_skip("input checksums", "no SHA256SUMS beside the prepared system"))
    else:
        import hashlib

        # Bounded: only the files this manifest names, all of them small preparation records.
        # Nothing is discovered by walking, and no production trajectory is ever hashed.
        bad = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, _, relative = line.partition("  ")
            path = inputs / relative
            if not path.is_file():
                bad.append(f"{relative} is missing")
                continue
            actual = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    actual.update(block)
            if actual.hexdigest() != digest:
                bad.append(f"{relative} changed since it was prepared")
        if bad:
            return results + [_no("input checksums", "; ".join(bad[:3]))]
        results.append(_ok("input checksums", "every prepared file matches SHA256SUMS"))
    return results


def check_forcefield(inputs):
    """`forcefield.json` and the resolved system configuration describe the same Hamiltonian.

    The record says what was BUILT and the configuration says what was REQUESTED. They can
    disagree, and when they do the trajectory belongs to the record, so a mismatch is refused here
    rather than discovered a year later.
    """
    record = inputs / "forcefield.json"
    resolved = inputs / "resolved_sys.config.yaml"
    if not record.is_file():
        return [_skip("force field", "no forcefield.json beside the prepared system")]
    try:
        built = json.loads(record.read_text(encoding="utf-8"))
        wanted = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except Exception as error:
        return [_no("force field", f"{type(error).__name__}: {error}")]

    solvation = built.get("solvation")
    if solvation != wanted.get("solvation"):
        return [_no("force field",
                    f"forcefield.json says solvation {solvation!r}, resolved config says "
                    f"{wanted.get('solvation')!r}")]

    requested = wanted.get("forcefield") or {}
    protein = (built.get("protein") or {})
    loaded = protein.get("openmm_resource") or protein.get("tleap_resource")
    if loaded and requested.get("protein") and loaded != requested["protein"]:
        return [_no("force field",
                    f"the System was built with {loaded!r} but the resolved configuration names "
                    f"{requested['protein']!r}")]
    water = (built.get("water") or {}).get("openmm_resource")
    if water and requested.get("water") and water != requested["water"]:
        return [_no("force field",
                    f"the System was built with water {water!r} but the resolved configuration "
                    f"names {requested['water']!r}")]
    described = loaded or "no protein force field"
    if water:
        described += f" + {water}"
    elif solvation == "implicit":
        described += f" + {(built.get('implicit_solvent') or {}).get('model')}"
    ligand = (built.get("ligand") or {}).get("openff_resource")
    if ligand:
        described += f" + {ligand}"
    return [_ok("force field", described)]


# ---------------------------------------------------------------------------------------------
# The platform
# ---------------------------------------------------------------------------------------------

def check_platform(*, dynamics=True, devices=None):
    """CUDA is available and the requested devices exist. There is no silent fallback.

    A run that quietly lands on the CPU still finishes and still writes a trajectory -- two orders
    of magnitude later, on a machine whose GPU was simply not visible to this process. So the
    fallback has to be asked for by name, and `--check` says which platform a real run would use.
    """
    from openmm import Platform

    available = sorted(Platform.getPlatform(i).getName()
                       for i in range(Platform.getNumPlatforms()))
    forced = os.environ.get("MD_PLATFORM")
    if forced:
        if forced not in available:
            return [_no("platform", f"MD_PLATFORM={forced} is not in this OpenMM build "
                                    f"({', '.join(available)})")]
        return [_ok("platform", f"{forced} (asked for by name via MD_PLATFORM)")]
    if "CUDA" not in available:
        return [_no("platform",
                    f"no CUDA platform in this OpenMM build (it has: {', '.join(available)}). "
                    f"Check the GPU is visible (nvidia-smi, CUDA_VISIBLE_DEVICES), or ask for "
                    f"another platform by name: MD_PLATFORM=CPU ./run.sh")]

    # The CUDA PLATFORM being listed is not the same as a CUDA DEVICE being visible. With
    # CUDA_VISIBLE_DEVICES="" OpenMM still offers the platform and a run dies at Context creation,
    # which is precisely the false green this check exists to prevent -- so zero devices fails
    # here, in `--check` as well as in a real run. A machine with no GPU is a supported machine;
    # it just has to ask for its platform by name (MD_PLATFORM=CPU).
    visible = _visible_cuda_devices()
    if not visible:
        return [_no("platform",
                    "OpenMM offers a CUDA platform but this process can see no CUDA device "
                    "(CUDA_VISIBLE_DEVICES is empty, or nvidia-smi lists none). A dynamics run "
                    "would fail when it created its Context. Make a device visible, or ask for "
                    "another platform by name: MD_PLATFORM=CPU ./run.sh")]
    results = [_ok("platform", f"CUDA, {len(visible)} device(s) visible")]

    if devices:
        outside = [d for d in devices if d not in visible]
        if outside:
            return results + [_no("cuda devices",
                                  f"requested device(s) {outside} are not visible; this process "
                                  f"sees {visible}")]
        results.append(_ok("cuda devices", f"{devices} within visible {visible}"))
    return results


def _visible_cuda_devices():
    import subprocess

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        return list(range(len([t for t in raw.split(",") if t.strip()])))
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return list(range(len([l for l in out.stdout.splitlines() if l.strip()])))
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------------------------

def check_parent(here, stage, *, dynamics=True):
    """The parent stage finished, and finished the request this one was generated against.

    `dynamics=False` is `--check` on a chain that has not started yet. A parent that simply has not
    run is PENDING there, not a failure: `run_all.sh --check` is meant to validate a fresh project
    before anything runs, and the stages are sequential by construction. A parent that HAS run and
    is inconsistent still fails either way -- that one is a real problem now.
    """
    # `parent` is null for the FIRST stage, which reads `common/initial_state.xml` rather than a
    # previous stage's handoff. Its input is covered by the prepared-inputs check, not here.
    if not stage or not stage.get("parent") or not stage.get("input_state"):
        return [_skip("parent stage", "this stage reads the prepared inputs")]
    parent_state = (here / stage["input_state"]).resolve()
    name = stage.get("parent") or "the parent stage"
    if not parent_state.is_file():
        pending = (f"{name} has not run yet; this stage runs after it")
        missing = (f"{name} has not written {Path(stage['input_state']).name}; run it first "
                   f"(cd {stage.get('parent_path') or '..'} && ./run.sh)")
        return [_skip("parent stage", pending) if not dynamics
                else _no("parent stage", missing)]
    record = parent_state.parent / "resolved_stage.yaml"
    if not record.is_file():
        return [_no("parent stage",
                    f"{name} has a final state but no resolved_stage.yaml, so it did not finish "
                    f"cleanly")]
    try:
        written = yaml.safe_load(record.read_text(encoding="utf-8")) or {}
    except Exception as error:
        return [_no("parent stage", f"{record.name} is unreadable ({type(error).__name__})")]
    if written.get("status") != "completed":
        return [_no("parent stage", f"{name} records status {written.get('status')!r}")]
    return [_ok("parent stage", f"{name} completed ({written.get('completed_steps', '?')} steps)")]


def check_own_completion(here, stage):
    """An existing completion record is internally consistent with the request it claims.

    Not a refusal to run: a consistent record means this stage is skipped, and an INCONSISTENT one
    means neither reusing nor overwriting is safe. That second case is what this catches.
    """
    final = here / (stage or {}).get("output_state", "final_state.xml")
    record = here / "resolved_stage.yaml"
    if not final.is_file() and not record.is_file():
        return [_ok("completion record", "absent; this stage has not run")]
    if final.is_file() != record.is_file():
        present, missing = ((final, record) if final.is_file() else (record, final))
        return [_no("completion record",
                    f"{present.name} exists but {missing.name} does not, so this stage did not "
                    f"finish cleanly")]
    try:
        written = yaml.safe_load(record.read_text(encoding="utf-8")) or {}
    except Exception as error:
        return [_no("completion record", f"unreadable ({type(error).__name__}: {error})")]
    if not written.get("stage_config_sha256"):
        return [_no("completion record",
                    "records no stage_config_sha256, so there is no way to tell which request "
                    "produced these outputs")]
    return [_ok("completion record", "present and self-consistent")]


# ---------------------------------------------------------------------------------------------
# Running the whole thing
# ---------------------------------------------------------------------------------------------

def run(here, project, inputs, config, *, stage=None, dynamics=True, devices=None, extra=None):
    """Every check, in order, as a list of rows. Nothing is written and nothing is integrated."""
    here, project, inputs = Path(here), Path(project), Path(inputs)
    results = []
    results += check_dataset(here, project, config)
    results += check_common(inputs, config)
    results += check_forcefield(inputs)
    results += check_platform(dynamics=dynamics, devices=devices)
    if stage is not None:
        results += check_parent(here, stage, dynamics=dynamics)
        results += check_own_completion(here, stage)
    for check in (extra or []):
        try:
            results += list(check())
        except Exception as error:                             # a check must not crash the run
            results.append(_no("check", f"{type(error).__name__}: {error}"))
    return results


def report(results, *, label="", stream=None):
    """A compact PASS/FAIL table. This is all `--check` prints."""
    stream = stream or sys.stdout
    width = max((len(r.name) for r in results), default=10)
    header = f"preflight{(' ' + label) if label else ''}"
    print(f"== {header} ==", file=stream)
    for result in results:
        detail = f"  {result.detail}" if result.detail else ""
        print(f"  [{result.status:4}] {result.name:<{width}}{detail}", file=stream)
    failed = [r for r in results if r.failed]
    print(f"  {len(results) - len(failed)}/{len(results)} ok"
          + (f", {len(failed)} FAILED" if failed else ""), file=stream)
    return not failed


def require(here, project, inputs, config, *, label="", **kwargs):
    """Run the checks and stop the process if any failed. Called before any Context exists."""
    results = run(here, project, inputs, config, **kwargs)
    if not report(results, label=label):
        raise SystemExit(
            f"preflight failed: {sum(1 for r in results if r.failed)} check(s). Nothing was run, "
            f"no Context was created, and no output was written.")
    return results
