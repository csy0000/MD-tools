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


def check_dataset(here, project, config, inputs=None):
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
    results += check_template_provenance(here, project, manifest=document, inputs=inputs,
                                         contract_managed=True)
    return results


#: Where a contract-managed project records the generator, and how to read it out of each.
#: Every one of these is REQUIRED when the project is contract-managed: a record that omits its
#: template identity used to be skipped, so deleting the field was enough to pass.
PROVENANCE_RECORDS = (
    ("dataset.yaml", "dataset.yaml", ("templates", "commit")),
    ("provenance.yaml", "provenance.yaml", ("template", "commit")),
    ("common provenance", "{inputs}/provenance.yaml", ("template", "commit")),
    ("resolved system config", "{inputs}/resolved_sys.config.yaml",
     ("provenance", "template", "commit")),
    ("md.config.yaml", "md.config.yaml", ("provenance", "template_commit")),
)


def _dig(document, keys):
    for key in keys:
        if not isinstance(document, dict):
            return None
        document = document.get(key)
    return document


def check_template_provenance(here, project, *, manifest=None, stage=None, inputs=None,
                              contract_managed=None):
    """Every record that should name the generator does, and they all name the same one.

    `dataset.yaml` says which MD-templates the dataset was generated by; the provenance files, the
    resolved configuration, each `stage.yaml` and each method record say which one wrote them. If a
    dataset's system was built from one checkout and its run scripts from another, the single
    recorded provenance is true of only half the tree -- and it is the half nobody checks that
    misleads.

    For a CONTRACT-MANAGED project every applicable record is required. Comparing only the fields
    that happen to be present meant a missing one was indistinguishable from an agreeing one.

    This compares RECORDS. Establishing what the generator actually was is `md_data_contract`'s job
    at generation time; a generated project deliberately cannot import `md_templates`, and nothing
    here inspects the original checkout.
    """
    project = Path(project)
    inputs = Path(inputs) if inputs is not None else None
    if manifest is None:
        path = project / "dataset.yaml"
        manifest = _read_yaml(path) if path.is_file() else None
    if contract_managed is None:
        contract_managed = manifest is not None
    if not contract_managed:
        return [_skip("template provenance",
                      "unregistered local project: no dataset.yaml, so there is no contract "
                      "identity to hold the other records to")]
    if manifest is None:
        return [_no("template provenance", "contract-managed, but dataset.yaml is unreadable")]

    seen, missing = {}, []
    for label, pattern, keys in PROVENANCE_RECORDS:
        relative = pattern.format(inputs=(inputs.name if inputs else "common"))
        path = (inputs / Path(relative).name if inputs and pattern.startswith("{inputs}")
                else project / relative)
        if not path.is_file():
            missing.append(f"{label} ({relative}) does not exist")
            continue
        document = _read_yaml(path)
        if document is None:
            missing.append(f"{label} is unreadable")
            continue
        value = _dig(document, keys)
        if not value:
            missing.append(f"{label} records no {'.'.join(keys)}")
            continue
        seen[label] = str(value).strip().lower()
        # A record carrying both routes must not disagree with itself.
        direct = _dig(document, ("implementation", "direct_url", "vcs_info", "commit_id"))
        checkout = _dig(document, ("implementation", "git_commit"))
        if direct and checkout and str(direct).lower() != str(checkout).lower():
            missing.append(f"{label} records git_commit {str(checkout)[:12]} and "
                           f"direct_url commit {str(direct)[:12]}, which disagree")

    # This stage, and the method record it belongs to, when there is one.
    if stage is None:
        stage_path = Path(here) / "stage.yaml"
        stage = _read_yaml(stage_path) if stage_path.is_file() else None
    if stage is not None:
        value = stage.get("template_commit")
        if not value:
            missing.append("this stage's stage.yaml records no template_commit")
        else:
            seen["stage.yaml"] = str(value).strip().lower()
    for name in ("path_definition.yaml", "rest2.yaml", "cmd.yaml"):
        path = Path(here) / name
        if not path.is_file():
            continue
        document = _read_yaml(path) or {}
        value = document.get("template_commit")
        if not value:
            missing.append(f"{name} records no template_commit")
        else:
            seen[name] = str(value).strip().lower()

    if missing:
        return [_no("template provenance",
                    f"contract-managed, so every generated record must name the generator: "
                    f"{'; '.join(missing[:3])}"
                    + (f" (and {len(missing) - 3} more)" if len(missing) > 3 else ""))]

    declared = seen.get("dataset.yaml")
    disagreeing = {where: value for where, value in seen.items() if value != declared}
    if disagreeing:
        detail = "; ".join(f"{where} says {value[:12]}"
                           for where, value in sorted(disagreeing.items()))
        return [_no("template provenance",
                    f"dataset.yaml records templates.commit {declared[:12]} but {detail}. The "
                    f"dataset and the scripts in it were not generated by the same MD-templates")]
    return [_ok("template provenance",
                f"{declared[:12]} agrees across {len(seen)} record(s)")]

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


def _exact(built, wanted, *, field, what, problems):
    """One consequential identity, compared exactly. An ABSENT expected value is a failure.

    The previous version compared only when both sides were present, so a record missing the field
    entirely passed -- and that is the case where what was built is least knowable.
    """
    if wanted in (None, ""):
        problems.append(f"the resolved configuration names no {what}, so there is nothing to "
                        f"check {field} against")
        return
    if built in (None, ""):
        problems.append(f"forcefield.json records no {field}, but the resolved configuration "
                        f"asked for {wanted!r}")
        return
    if built != wanted:
        problems.append(f"{field} was built as {built!r} but the resolved configuration names "
                        f"{wanted!r}")


def _absent(built, *, field, why, problems):
    """A force field that must NOT have been loaded for this route."""
    if built not in (None, "", [], {}):
        problems.append(f"forcefield.json claims {field} {built!r}, but {why}")


def _sha256_file(path):
    """SHA-256 of one SMALL named file. Never used on a trajectory."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_forcefield(inputs):
    """`forcefield.json` and the resolved system configuration describe the same Hamiltonian.

    The record says what was BUILT and the configuration says what was REQUESTED. They can
    disagree, and when they do the trajectory belongs to the record, so a mismatch is refused here
    rather than discovered a year later.

    Route-aware: a peptide system must not claim a ligand force field, a ligand system must not
    claim a protein one, and an implicit system must claim neither a water model nor a barostat.
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
    if solvation not in ("explicit", "implicit"):
        return [_no("force field", f"solvation {solvation!r} is neither explicit nor implicit")]

    requested = wanted.get("forcefield") or {}
    solute = wanted.get("solute") or {}
    protein = built.get("protein") or {}
    ligand = built.get("ligand") or {}
    water = built.get("water") or {}
    implicit = built.get("implicit_solvent") or {}

    # The route is decided by what the record says was actually parameterised, and cross-checked
    # against what the configuration asked for. Disagreement about the ROUTE is itself a failure.
    is_ligand = bool(ligand.get("openff_resource"))
    wants_ligand = not bool(solute.get("peptide", True))
    if is_ligand != wants_ligand:
        return [_no("force field",
                    f"forcefield.json describes {'a ligand' if is_ligand else 'a peptide'} route "
                    f"but the resolved configuration asks for "
                    f"{'a ligand' if wants_ligand else 'a peptide'} one")]

    problems, described = [], []
    if is_ligand:
        _exact(ligand.get("openff_resource"), requested.get("ligand"),
               field="the OpenFF resource", what="ligand force field", problems=problems)
        _exact(ligand.get("charge_method"), requested.get("ligand_charge_method"),
               field="the ligand charge method", what="ligand charge method", problems=problems)
        _absent(protein.get("openmm_resource") or protein.get("tleap_resource"),
                field="protein force field",
                why="this is a ligand system and no protein was parameterised", problems=problems)
        described.append(str(ligand.get("openff_resource")))
        described.append(f"charges {ligand.get('charge_method')}")
    else:
        _exact(protein.get("openmm_resource") or protein.get("tleap_resource"),
               requested.get("protein"), field="the protein force field",
               what="protein force field", problems=problems)
        _absent(ligand.get("openff_resource"), field="ligand force field",
                why="this is a peptide system and no ligand was parameterised", problems=problems)
        described.append(str(protein.get("openmm_resource") or protein.get("tleap_resource")))

    if solvation == "explicit":
        _exact(water.get("openmm_resource"), requested.get("water"),
               field="the water model", what="water model", problems=problems)
        described.append(str(water.get("openmm_resource")))
    else:
        # No box, so no water and no barostat. Claiming either would mean the record describes a
        # System that is not the one this route builds.
        _absent(water.get("openmm_resource"), field="a water model",
                why="this is an implicit-solvent system with no box", problems=problems)
        wanted_implicit = wanted.get("implicit_solvent") or {}
        _exact(implicit.get("model"), wanted_implicit.get("model"),
               field="the implicit model", what="implicit solvent model", problems=problems)
        _exact(implicit.get("radii"), wanted_implicit.get("radii"),
               field="the radius set", what="implicit radius set", problems=problems)
        if bool(implicit.get("nonpolar_sasa")) != bool(wanted_implicit.get("nonpolar_sasa")):
            problems.append(
                f"the ACE nonpolar surface-area term was built "
                f"{'on' if implicit.get('nonpolar_sasa') else 'off'} but the resolved "
                f"configuration asks for it "
                f"{'on' if wanted_implicit.get('nonpolar_sasa') else 'off'}")
        if (built.get("nonbonded") or {}).get("barostat"):
            problems.append("forcefield.json claims a barostat, but an implicit-solvent System "
                            "has no box to scale")
        described.append(f"{implicit.get('model')}/{implicit.get('radii')}")

    if problems:
        return [_no("force field", "; ".join(problems))]
    return [_ok("force field", " + ".join(d for d in described if d and d != "None"))]


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

def _fingerprint(document):
    """The canonical stage signature, from the generated project's own implementation.

    Imported rather than reimplemented. Two subtly different hashes over "the stage request" is
    exactly the failure this check exists to catch: the run would write one and the check would
    compare another, and every stage would look stale or every stale stage would look fine.
    """
    from md_stages import stage_config_sha256

    return stage_config_sha256(document)


def _read_yaml(path):
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def check_parent(here, stage, *, dynamics=True):
    """The parent stage finished, and finished the request THIS one was generated against.

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

    directory = parent_state.parent
    record_path = directory / "resolved_stage.yaml"
    if not record_path.is_file():
        return [_no("parent stage",
                    f"{name} has a final state but no resolved_stage.yaml, so it did not finish "
                    f"cleanly")]
    written = _read_yaml(record_path)
    if written is None:
        return [_no("parent stage", f"{record_path.name} is unreadable")]
    if written.get("status") != "completed":
        return [_no("parent stage", f"{name} records status {written.get('status')!r}")]

    # The parent's request as it stands NOW, recomputed. A stored hash being present says only
    # that something was recorded; it says nothing about whether the parent still asks for what it
    # asked for when it ran.
    parent_stage = _read_yaml(directory / "stage.yaml")
    if parent_stage is None:
        return [_no("parent stage", f"{name} has no readable stage.yaml to check its record "
                                    f"against")]
    recorded = written.get("stage_config_sha256")
    if not recorded:
        return [_no("parent stage",
                    f"{name} records no stage_config_sha256, so there is no way to tell which "
                    f"request produced its final state")]
    current = _fingerprint(parent_stage)
    if current != recorded:
        return [_no("parent stage",
                    f"{name}/stage.yaml has changed since it ran: it now hashes to "
                    f"{current[:12]} but its completion record was written for {recorded[:12]}. "
                    f"This stage would start from a state produced by a different request; rerun "
                    f"{name} or restore its configuration")]

    # The handoff itself. `final_state.xml` is small and is the one artifact this stage consumes,
    # so its recorded digest is compared against the bytes actually on disk.
    outputs = written.get("outputs") or {}
    handoff = (outputs.get(Path(stage["input_state"]).name)
               or outputs.get(written.get("output_state") or "final_state.xml") or {})
    expected = handoff.get("sha256")
    if expected:
        actual = _sha256_file(parent_state)
        if actual != expected:
            return [_no("parent stage",
                        f"{name}'s {parent_state.name} has changed since it was recorded "
                        f"({actual[:12]} on disk, {expected[:12]} recorded). The state this stage "
                        f"would start from is not the one {name} produced")]
    identity = "handoff verified" if expected else "handoff digest not recorded"
    return [_ok("parent stage",
                f"{name} completed ({written.get('completed_steps', '?')} steps), request "
                f"unchanged, {identity}")]


def _check_production_consistency(here, stage):
    """A production stage's recorded request still matches the one being asked for.

    Consistency ONLY. Whether the stage finished, and whether it should run again, is the
    launcher's own business against `resolved_run.yaml`: a common stage is fixed at generation and
    refuses to rerun, but production may legitimately be asked to run longer from its own
    checkpoint, and refusing that here would break the documented extension path in order to catch
    an unsafe edit. The unsafe edit is caught by the invariant comparison below and by
    `check_runtime_request`.
    """
    record = here / "resolved_stage.yaml"
    if not record.is_file():
        return [_ok("completion record", "absent; this stage has not run")]
    written = _read_yaml(record)
    if written is None:
        return [_no("completion record", "unreadable")]
    recorded = written.get("stage_invariant_sha256")
    if not recorded:
        return [_no("completion record",
                    "a production stage's record must carry stage_invariant_sha256, so that a "
                    "continuation can be checked against the physics that produced its outputs; "
                    "this one does not. It was written by an older generator -- regenerate the "
                    "stage rather than continuing into it")]
    current = _md_stages().stage_invariant_sha256(stage or {})
    if current != recorded:
        return [_no("completion record",
                    f"the stage request has changed since this stage last ran: its invariants now "
                    f"hash to {current[:12]} but the record was written for {recorded[:12]}. "
                    f"Continuing would apply new physics to output produced under the old "
                    f"request. Asking for a longer run alone is allowed and does not change this "
                    f"hash")]
    return [_ok("completion record", f"consistent with the recorded request ({recorded[:12]})")]


def check_own_completion(here, stage):
    """An existing completion record is consistent with the request it claims AND with this one.

    Not a refusal to run in general: a consistent record means this stage is skipped, and an
    INCONSISTENT one means neither reusing nor overwriting is safe. That second case is what this
    catches -- including a completed stage whose stage.yaml has since been edited, which a
    nonempty hash string alone would happily accept.
    """
    if (stage or {}).get("production"):
        return _check_production_consistency(here, stage)

    output_state = (stage or {}).get("output_state", "final_state.xml")
    if output_state is None:
        # A stage that declares no single handoff state -- REST2 keeps one per replica. There is
        # nothing here to compare, and pretending otherwise would report it as never having run.
        return [_skip("completion record", "this stage has no single handoff state")]
    final = here / output_state
    record = here / "resolved_stage.yaml"
    if not final.is_file() and not record.is_file():
        return [_ok("completion record", "absent; this stage has not run")]
    if final.is_file() != record.is_file():
        present, missing = ((final, record) if final.is_file() else (record, final))
        return [_no("completion record",
                    f"{present.name} exists but {missing.name} does not, so this stage did not "
                    f"finish cleanly")]
    written = _read_yaml(record)
    if written is None:
        return [_no("completion record", "unreadable")]
    recorded = written.get("stage_config_sha256")
    if not recorded:
        return [_no("completion record",
                    "records no stage_config_sha256, so there is no way to tell which request "
                    "produced these outputs")]
    if written.get("status") != "completed":
        return [_no("completion record",
                    f"a final state exists but the record says status "
                    f"{written.get('status')!r}, so this stage did not finish")]

    # A PRODUCTION stage may legitimately be asked to run longer, and running longer changes the
    # whole-document hash. Compare its invariants instead, which is exactly the set an extension
    # may not touch. A common stage is fixed at generation, so its whole document still counts.
    if (stage or {}).get("production"):
        recorded_invariant = written.get("stage_invariant_sha256")
        if not recorded_invariant:
            return [_no("completion record",
                        "a production completion record must carry stage_invariant_sha256, so "
                        "that a continuation can be checked against the physics that produced "
                        "these outputs; this one does not")]
        current = _md_stages().stage_invariant_sha256(stage or {})
        if current != recorded_invariant:
            return [_no("completion record",
                        f"the stage request has changed since this stage ran: its invariants now "
                        f"hash to {current[:12]} but the completion record was written for "
                        f"{recorded_invariant[:12]}. Continuing would apply new physics to output "
                        f"produced under the old request -- generate into a new output directory, "
                        f"or restore the request. Asking for a longer run alone is allowed")]
    else:
        current = _fingerprint(stage or {})
        if current != recorded:
            return [_no("completion record",
                        f"stage.yaml has changed since this stage ran: it now hashes to "
                        f"{current[:12]} but the completion record was written for "
                        f"{recorded[:12]}. Neither reusing these outputs nor overwriting them is "
                        f"safe -- generate into a new output directory, or remove this stage's "
                        f"outputs deliberately")]

    outputs = written.get("outputs") or {}
    expected = (outputs.get(final.name) or {}).get("sha256")
    if expected:
        actual = _sha256_file(final)
        if actual != expected:
            return [_no("completion record",
                        f"{final.name} has changed since it was recorded ({actual[:12]} on disk, "
                        f"{expected[:12]} recorded)")]
    return [_ok("completion record",
                f"complete and unchanged ({recorded[:12]}"
                f"{', final state verified' if expected else ''})")]


# ---------------------------------------------------------------------------------------------
# Running the whole thing
# ---------------------------------------------------------------------------------------------

def check_runtime_request(here, stage, config):
    """The configuration on disk NOW still describes the stage this directory was generated for.

    Generated launchers reread `md.config.yaml` at run time, which is what makes extension work:
    raise the requested length and the stage runs longer from its own checkpoint. The same
    reread is what let an edited timestep through -- `md-gen` refuses 4 fs without hydrogen mass
    repartitioning, but nothing re-applied that at run time, so editing the file after generation
    and rerunning appended dynamics at the new timestep to a trajectory produced at the old one.

    So the comparison is by INVARIANT fingerprint, not by whole document: recompute the request
    from the current configuration and compare it against the one `stage.yaml` recorded. The
    extendable fields are excluded, so a longer run still passes; everything else -- timestep,
    temperature, ensemble, tau, barostat, solvent mode, hydrogen mass, the parent handoff -- fails
    the moment it differs, before a Context exists.
    """
    if not stage or not stage.get("production"):
        return []                                   # common stages are fixed at generation
    method = stage.get("name")
    if not method or method not in (config or {}):
        return [_no("runtime request",
                    f"stage.yaml names method {method!r}, which the current md.config.yaml does "
                    f"not define, so what would run cannot be compared with what was generated")]
    try:
        helpers = _md_stages()
        current = helpers.production_stage_document(
            config, method,
            parent_stage=stage.get("parent"), parent_path=stage.get("parent_path"),
            implicit=bool(stage.get("implicit")),
            seeds={"integrator": stage.get("integrator_seed"),
                   "velocities": stage.get("velocity_seed"),
                   "barostat": stage.get("barostat_seed")},
            template_commit=stage.get("template_commit"),
            omega_excluded=stage.get("omega_excluded_bonds") or (),
        )
        recorded = helpers.stage_invariant_sha256(stage)
        now = helpers.stage_invariant_sha256(current)
    except Exception as error:
        return [_no("runtime request", f"{type(error).__name__}: {error}")]
    if recorded == now:
        return [_ok("runtime request", "the configuration still describes this stage")]

    changed = sorted(
        key for key in set(stage) | set(current)
        if key not in helpers.PRODUCTION_EXTENDABLE_FIELDS and stage.get(key) != current.get(key))
    detail = ", ".join(f"{k}: {stage.get(k)!r} -> {current.get(k)!r}" for k in changed) or \
        "the resolved request differs"
    return [_no("runtime request",
                f"md.config.yaml no longer describes the stage this directory was generated for "
                f"({detail}). Continuing would apply the new setting to output produced under the "
                f"old one. Regenerate into a new stage, or restore the configuration; raising the "
                f"requested length alone is a legitimate extension and is allowed.")]


def _md_stages():
    """The project's own copy of md_stages, beside md.config.yaml above this stage."""
    import importlib.util

    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        module_path = candidate / "md_stages.py"
        if module_path.is_file():
            spec = importlib.util.spec_from_file_location("_preflight_md_stages", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("md_stages.py was not found beside or above preflight.py")


def run(here, project, inputs, config, *, stage=None, dynamics=True, devices=None, extra=None):
    """Every check, in order, as a list of rows. Nothing is written and nothing is integrated."""
    here, project, inputs = Path(here), Path(project), Path(inputs)
    results = []
    results += check_dataset(here, project, config, inputs=inputs)
    results += check_common(inputs, config)
    results += check_forcefield(inputs)
    results += check_platform(dynamics=dynamics, devices=devices)
    if stage is not None:
        results += check_parent(here, stage, dynamics=dynamics)
        results += check_own_completion(here, stage)
        results += check_runtime_request(here, stage, config)
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
