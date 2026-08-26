#!/usr/bin/env python
"""Read an MD-templates 0.3.x simulation directory and write a FAIR registration candidate beside it.

    python scripts/retrofit_fair_v030.py \\
        --inputs /path/to/inputs --md /path/to/MD --output /path/to/fair-registration \\
        [--original-input /path/to/original.pdb] [--environment /path/to/environment-record]

Needs Python and PyYAML. No OpenMM, no CUDA, no network, no source checkout.

The directories are READ-ONLY. Nothing here opens a source file for writing, renames one, touches a
timestamp or adds a file inside `inputs/` or `MD/`. Everything produced goes to `--output`, which
must not already exist or must be empty.

What this cannot do is invent history. A 0.3.x directory DID record the original input's SHA-256
in `inputs/provenance.yaml` when one was available -- but it generally did not retain the original
bytes, and it recorded neither the build environment nor an exact code identity. No amount of
reading the tree recovers those. Every retrospective value therefore carries an evidence status -- `recorded`, `derived`,
`user_supplied` or `unknown` -- and `unknown` is a legitimate, final answer. There is deliberately
no `inferred`: a guess dressed as a scientific value is worse than a gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

FORMAT = "md-templates-fair-retrofit/v1"
#: The sidecar's own manifests cannot appear in the manifest they define.
SELF_EXCLUDED = ("SHA256SUMS",)
PROGRESS_BYTES = 256 * 1024 * 1024


def sha256_file(path: Path, *, label: str = "") -> str:
    """Streamed, with progress for files big enough that silence looks like a hang."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    done = 0
    announce = size >= PROGRESS_BYTES
    if announce:
        print(f"    hashing {label or path.name} ({size / 1e9:.2f} GB)", end="", flush=True)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
            done += len(chunk)
            if announce and done % (1 << 30) < (1 << 22):
                print(".", end="", flush=True)
    if announce:
        print(" done", flush=True)
    return digest.hexdigest()


def evidence(value: Any, status: str, *, source: Optional[str] = None) -> dict[str, Any]:
    """A value and how it is known. `unknown` values keep their status and stay null."""
    assert status in ("recorded", "derived", "user_supplied", "unknown"), status
    return {"value": value, "evidence": status, "source": source}


def load_yaml(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:                                   # noqa: BLE001
        return None


def inventory(root: Path, *, exclude_dirs: tuple[Path, ...] = ()) -> list[dict[str, Any]]:
    """Every regular file under `root`, with size, mtime and digest. Read-only."""
    entries = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        # Path-aware, not string-prefix: `/x/out` must not exclude `/x/outputs`, which a
        # startswith() check silently would.
        if any(path.is_relative_to(d) for d in exclude_dirs):
            continue
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        entries.append({"path": relative, "bytes": stat.st_size,
                        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "sha256": sha256_file(path, label=relative)})
    return entries


def snapshot(root: Path, base: Path) -> dict[str, tuple]:
    """Path, size, mtime and digest for every file, for a real before/after comparison."""
    return {path.relative_to(base).as_posix():
            (path.stat().st_size, path.stat().st_mtime_ns, sha256_file(path))
            for path in sorted(root.rglob("*")) if path.is_file() and not path.is_symlink()}


def classify(system: dict, run: dict, has_original: bool) -> dict[str, Any]:
    """A, B or C, with every reason that lowered it, machine-readable.

    Grade A is a strong claim -- "this can be rebuilt from the original input" -- so it needs the
    original input retained AND verified, an EXACT code identity, the build environment, and the
    runtime records the configured protocol actually requires. A package version like `0.3.1` is
    not exact identity: it names a release, not a build, and two builds of the same version can
    differ. A grade is never raised because a directory name looks right.
    """
    reasons = []
    prepared = all(system["payload_present"].get(name) for name in
                   ("system.xml", "topology.pdb", "initial_state.xml"))
    if not prepared:
        reasons.append({"code": "prepared_system_incomplete",
                        "detail": "system.xml, topology.pdb or initial_state.xml is missing"})
    if not system["resolved_config_present"]:
        reasons.append({"code": "resolved_system_config_missing",
                        "detail": "resolved_sys.config.yaml is absent"})
    if not run["md_config_present"]:
        reasons.append({"code": "md_config_missing", "detail": "MD/md.config.yaml is absent"})
    if not has_original:
        reasons.append({"code": "original_input_absent",
                        "detail": "the original molecular input was not supplied and verified, so "
                                  "parameterisation cannot be rebuilt from source"})

    # Exact identity: a recorded commit or an installed fingerprint. Not a version string.
    identity = system["implementation"]["value"] or {}
    exact = bool(identity.get("git_commit") or identity.get("installed_fingerprint"))
    if not exact:
        reasons.append({"code": "implementation_identity_not_exact",
                        "detail": "no git_commit and no installed_fingerprint were recorded; a "
                                  "package version alone names a release, not the build that ran"})
    if not system["environment_known"]:
        reasons.append({"code": "build_environment_unknown",
                        "detail": "preparation package versions were not recorded and were not "
                                  "supplied"})

    for gap in run.get("runtime_gaps", []):
        reasons.append(gap)

    blocking = {"prepared_system_incomplete", "resolved_system_config_missing",
                "md_config_missing"}
    if any(r["code"] in blocking for r in reasons):
        grade = "C"
    elif not reasons:
        grade = "A"
    else:
        grade = "B"
    return {"grade": grade,
            "meaning": {"A": "rebuildable from the original molecular input",
                        "B": "prepared-system reproducible; parameterisation, code identity, "
                             "environment or runtime closure incomplete",
                        "C": "archival/analysis only"}[grade],
            "reasons": reasons}


def runtime_gaps(md: Path, md_config: Optional[dict]) -> list[dict[str, str]]:
    """Which records the CONFIGURED protocol requires and this directory does not have.

    The protocol is read from md.config.yaml. If it cannot be read, that is itself the gap -- the
    requirement is never guessed from folder names, because a folder called `cMD` proves only that
    somebody made a folder.
    """
    if not md_config:
        return [{"code": "protocol_undeterminable",
                 "detail": "MD/md.config.yaml could not be read, so the records this run should "
                           "have cannot be determined; the protocol is not inferred from folders"}]
    gaps = []
    methods = [str(m) for m in (md_config.get("methods") or [])]
    if not methods:
        gaps.append({"code": "protocol_undeterminable",
                     "detail": "md.config.yaml lists no methods"})
    for stage_dir in ("minimization",):
        if not (md / stage_dir / "resolved_stage.yaml").is_file():
            gaps.append({"code": "common_stage_record_missing",
                         "detail": f"{stage_dir}/resolved_stage.yaml is absent"})
    eq = md / "eq"
    if eq.is_dir() and not any(eq.rglob("resolved_stage.yaml")):
        gaps.append({"code": "common_stage_record_missing",
                     "detail": "no equilibration stage recorded a resolved_stage.yaml"})
    for method in methods:
        if method == "cMD" and not any((md / "cMD").glob("resolved_*.yaml")):
            gaps.append({"code": "cmd_runtime_record_missing",
                         "detail": "md.config.yaml configures cMD but MD/cMD has no resolved run "
                                   "record"})
        if method == "REST2" and not (md / "REST2" / "resolved_run.yaml").is_file():
            gaps.append({"code": "rest2_runtime_record_missing",
                         "detail": "md.config.yaml configures REST2 but MD/REST2 has no "
                                   "resolved_run.yaml"})
    return gaps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="retrofit_fair_v030.py",
        description="Write a FAIR registration candidate for a 0.3.x inputs/ + MD/ pair. "
                    "Reads only; never modifies the source.")
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--md", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--original-input", type=Path, default=None,
                        help="the original molecular input, if you still have it. Accepted only "
                             "when its SHA-256 matches the hash 0.3.x recorded.")
    parser.add_argument("--environment", type=Path, default=None,
                        help="a YAML/JSON record of the build environment, if you have one")
    args = parser.parse_args(argv)

    inputs, md, out = args.inputs.resolve(), args.md.resolve(), args.output.resolve()
    for name, path in (("--inputs", inputs), ("--md", md)):
        if not path.is_dir():
            print(f"error: {name} {path} is not a directory", file=sys.stderr)
            return 2
    if out.exists() and any(out.iterdir()):
        print(f"error: --output {out} is not empty. Refusing to write into a directory that "
              f"already holds files; choose a new one.", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    print(f"reading  inputs: {inputs}")
    print(f"reading  MD    : {md}")
    print(f"writing  output: {out}")

    sys_prov = load_yaml(inputs / "provenance.yaml") or {}
    sys_resolved = load_yaml(inputs / "resolved_sys.config.yaml")
    md_prov = load_yaml(md / "provenance.yaml") or {}
    md_config = load_yaml(md / "md.config.yaml")

    # --- the original molecular input -------------------------------------------------------
    recorded_hashes = ((sys_prov.get("generated") or {}).get("input_hashes")
                       or {})
    original_status, original_record = "unknown", None
    if args.original_input is not None:
        if not args.original_input.is_file():
            print(f"error: --original-input {args.original_input} does not exist", file=sys.stderr)
            return 2
        supplied = sha256_file(args.original_input)
        expected = recorded_hashes.get(args.original_input.name)
        if expected is None:
            print(f"error: 0.3.x recorded no hash for '{args.original_input.name}'. Recorded "
                  f"names: {sorted(recorded_hashes) or 'none'}. Refusing to accept an input that "
                  f"cannot be checked.", file=sys.stderr)
            return 3
        if supplied != expected:
            print(f"error: --original-input SHA-256 mismatch\n  supplied: {supplied}\n  "
                  f"recorded: {expected}\nThis is not the file this system was built from.",
                  file=sys.stderr)
            return 3
        original_status = "user_supplied"
        original_record = {"name": args.original_input.name, "sha256": supplied,
                           "verified_against": "inputs/provenance.yaml",
                           # Filled in below once the output directory exists: a verified original
                           # that is not retained leaves the candidate un-rebuildable, which is the
                           # whole distinction between grade A and grade B.
                           "retained_path": None}
        print(f"  original input verified against the recorded hash: {supplied[:16]}...")
    elif recorded_hashes:
        original_status = "recorded"
        original_record = {"name": next(iter(recorded_hashes)),
                           "sha256": next(iter(recorded_hashes.values())),
                           "note": "hash recorded by 0.3.x; the file itself is not present"}

    supplied_environment = load_yaml(args.environment) if args.environment else None

    # Retain the evidence that was supplied, so the candidate is self-contained.
    if original_status == "user_supplied":
        kept = out / "original_inputs" / args.original_input.name
        kept.parent.mkdir(parents=True, exist_ok=True)
        if kept.exists() and sha256_file(kept) != original_record["sha256"]:
            print(f"error: {kept} already exists with different content", file=sys.stderr)
            return 2
        shutil.copy2(args.original_input, kept)
        original_record["retained_path"] = kept.relative_to(out).as_posix()
        print(f"  retained the original input at {original_record['retained_path']}")
    environment_source = None
    if supplied_environment is not None:
        retained = out / "environment" / args.environment.name
        retained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.environment, retained)
        # The retained relative path, never the absolute one it came from.
        environment_source = retained.relative_to(out).as_posix()

    # --- inventory, read-only ----------------------------------------------------------------
    print("  hashing source files (read-only)")
    common_root = Path(os.path.commonpath([str(inputs), str(md)]))
    before_snapshot = {**snapshot(inputs, common_root), **snapshot(md, common_root)}
    exclude = (out,) if out.is_relative_to(common_root) else ()
    files = []
    for root in (inputs, md):
        for entry in inventory(root, exclude_dirs=exclude):
            files.append({**entry,
                          "path": (root / entry["path"]).relative_to(common_root).as_posix()})

    system_record = {
        "format": FORMAT,
        "record_kind": "system",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_inputs": inputs.relative_to(common_root).as_posix(),
        "original_input": evidence(original_record, original_status,
                                   source="inputs/provenance.yaml"),
        "resolved_system_config": evidence(sys_resolved, "recorded" if sys_resolved else "unknown",
                                           source="inputs/resolved_sys.config.yaml"),
        "implementation": evidence((sys_prov.get("md_templates") or None),
                                   "recorded" if sys_prov.get("md_templates") else "unknown",
                                   source="inputs/provenance.yaml"),
        "environment": evidence(supplied_environment,
                                "user_supplied" if supplied_environment else "unknown",
                                source=environment_source),
        "payload_present": {name: (inputs / name).is_file() for name in
                            ("system.xml", "topology.pdb", "solute.pdb", "initial_state.xml",
                             "solute.yaml", "resolved_sys.config.yaml", "provenance.yaml")},
        "resolved_config_present": sys_resolved is not None,
        "environment_known": supplied_environment is not None,
    }

    run_record = {
        "format": FORMAT,
        "record_kind": "run",
        "created_utc": system_record["created_utc"],
        "source_md": md.relative_to(common_root).as_posix(),
        "resolved_md_config": evidence(md_config, "recorded" if md_config else "unknown",
                                       source="MD/md.config.yaml"),
        "md_generation_provenance": evidence(md_prov or None,
                                             "recorded" if md_prov else "unknown",
                                             source="MD/provenance.yaml"),
        "stages": _stage_records(md),
        "md_config_present": md_config is not None,
        "runtime_gaps": runtime_gaps(md, md_config),
        # A replay suggestion, never presented as what was actually run: 0.3.x did not record the
        # command, and labelling a reconstruction as the original would be a fabricated fact.
        "reconstructed_command": evidence(
            ["md-openmm", "md-gen", "-if", "./inputs/", "--config", "md.config.yaml",
             "-of", "./MD/"], "derived",
            source="reconstructed from the 0.3.x layout; the original command was not recorded"),
    }

    forcefield = _forcefield_from_resolved(sys_resolved)
    grade = classify(system_record, run_record, has_original=original_status == "user_supplied")

    (out / "system-record.yaml").write_text(yaml.safe_dump(system_record, sort_keys=False),
                                            encoding="utf-8")
    (out / "run-record.yaml").write_text(yaml.safe_dump(run_record, sort_keys=False),
                                         encoding="utf-8")
    (out / "forcefield.json").write_text(json.dumps(forcefield, indent=2) + "\n", encoding="utf-8")
    # No absolute common_root: the manifest documents the root by describing it, not by baking in
    # a path that stops being true the moment the tree is archived elsewhere.
    (out / "file-inventory.yaml").write_text(
        yaml.safe_dump({"format": FORMAT,
                        "paths_relative_to": "the common parent of inputs/ and MD/",
                        "inputs_dir": inputs.relative_to(common_root).as_posix(),
                        "md_dir": md.relative_to(common_root).as_posix(),
                        "file_count": len(files), "files": files}, sort_keys=False),
        encoding="utf-8")

    # Compare real snapshots rather than asserting a literal. "source_modified: false" written
    # unconditionally is a claim, not a check -- and a claim in a validation file is worse than
    # no claim, because it is the field a reader trusts.
    after_snapshot = {**snapshot(inputs, common_root), **snapshot(md, common_root)}
    changed = sorted(k for k in before_snapshot.keys() & after_snapshot.keys()
                     if before_snapshot[k] != after_snapshot[k])
    added = sorted(after_snapshot.keys() - before_snapshot.keys())
    removed = sorted(before_snapshot.keys() - after_snapshot.keys())

    validation = {
        "format": FORMAT,
        "created_utc": system_record["created_utc"],
        "classification": grade,
        "source_verification": {
            "method": "path, size, mtime_ns and SHA-256 compared before and after sidecar creation",
            "files_before": len(before_snapshot),
            "files_after": len(after_snapshot),
            "counts_match": len(before_snapshot) == len(after_snapshot),
            "changed": changed,
            "added": added,
            "removed": removed,
            "unmodified": not (changed or added or removed),
        },
        "source_file_count": len(files),
        "evidence_counts": _evidence_counts(system_record, run_record),
        "not_performed": [
            "no MD-data dataset identifier was assigned",
            "nothing was copied into $MD_DATA",
            "no source file was modified, moved or deleted",
            "no archive checksum manifest for the final dataset was produced",
            "no release tag was created or moved",
        ],
        "handoff_required_from_md_data": [
            "permanent dataset ID and landing page",
            "immutable storage location and access policy",
            "complete archival checksum manifest including trajectories",
            "retention, replacement and lifecycle records",
        ],
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    _write_readme(out, grade, common_root, inputs, md)

    # Sidecar manifest last: covers the source tree plus the sidecar payload, minus itself.
    # Every entry resolves from ONE documented root: the common project root. The sidecar's own
    # prefix is its real directory name, computed -- hardcoding "fair-registration/" broke every
    # path the moment anyone passed --output out.
    lines = [f"{entry['sha256']}  {entry['path']}" for entry in files]
    if out.is_relative_to(common_root):
        prefix = out.relative_to(common_root).as_posix()
    else:
        prefix = out.name
        print(f"  note: --output is outside the project root; sidecar entries are prefixed "
              f"'{prefix}/' and resolve from {out.parent}")
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name in SELF_EXCLUDED:
            continue
        lines.append(f"{sha256_file(path)}  {prefix}/{path.relative_to(out).as_posix()}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_root = common_root if out.is_relative_to(common_root) else out.parent

    print(f"\nclassification: {grade['grade']} -- {grade['meaning']}")
    for reason in grade["reasons"]:
        print(f"  - {reason['code']}: {reason['detail']}")
    print(f"\nwrote {len(list(out.iterdir()))} files to {out}")
    print("This is a registration candidate. MD-data assigns the dataset ID and archives it.")
    return 0


def _stage_records(md: Path) -> list[dict[str, Any]]:
    stages = []
    for path in sorted(md.rglob("resolved_stage.yaml")):
        stages.append({"path": path.parent.relative_to(md).as_posix(),
                       "record": load_yaml(path)})
    for name in ("cMD/resolved_run.yaml", "REST2/resolved_run.yaml"):
        record = load_yaml(md / name)
        if record is not None:
            stages.append({"path": name, "record": record})
    return stages


def _forcefield_from_resolved(resolved: Optional[dict]) -> dict[str, Any]:
    """The force-field record 0.3.x never wrote, rebuilt only from values it did record."""
    if not resolved:
        return {"format": FORMAT, "record_kind": "forcefield",
                "note": "resolved_sys.config.yaml is absent; nothing can be stated",
                "evidence": "unknown"}
    forcefield = resolved.get("forcefield") or {}
    implicit = resolved.get("solvation") == "implicit"
    solvent = resolved.get("solvent") or {}
    return {
        "format": FORMAT, "record_kind": "forcefield", "evidence": "recorded",
        "source": "inputs/resolved_sys.config.yaml",
        "solvation": resolved.get("solvation"),
        "protein": {"openmm_resource": forcefield.get("protein")},
        "water": {"model": solvent.get("model") if not implicit else None,
                  "openmm_resource": forcefield.get("water")},
        "implicit_solvent": resolved.get("implicit_solvent") if implicit else None,
        "explicit_solvent": None if implicit else solvent,
        "constraints": resolved.get("constraints"),
        "solute": resolved.get("solute"),
        "package_versions": {"evidence": "unknown",
                             "note": "0.3.x did not record preparation package versions"},
    }


def _evidence_counts(*records: dict) -> dict[str, int]:
    counts: dict[str, int] = {}

    def walk(node):
        if isinstance(node, dict):
            status = node.get("evidence")
            if isinstance(status, str) and status in ("recorded", "derived", "user_supplied",
                                                      "unknown"):
                counts[status] = counts.get(status, 0) + 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for record in records:
        walk(record)
    return counts


def _write_readme(out: Path, grade: dict, common_root: Path, inputs: Path, md: Path) -> None:
    (out / "README.md").write_text(f"""# FAIR registration candidate

Produced by `scripts/retrofit_fair_v030.py` from an MD-templates 0.3.x simulation directory.

    inputs : {inputs.relative_to(common_root).as_posix()}
    MD     : {md.relative_to(common_root).as_posix()}

**Classification {grade['grade']} — {grade['meaning']}.**

{chr(10).join('- `' + r['code'] + '`: ' + r['detail'] for r in grade['reasons']) or '- no reasons lowered this grade'}

## What this is, and is not

The source directories were read and not modified. Nothing here assigns a dataset identifier,
copies data into `$MD_DATA`, claims immutability or deletes anything: those belong to MD-data.

Every retrospective value carries an evidence status. `unknown` means the 0.3.x output did not
record it and it was not supplied — it is a final answer, not a placeholder. There is no
`inferred` status, because a guessed scientific value is worse than an acknowledged gap.

## Files

| file | contents |
|---|---|
| `system-record.yaml` | prepared system, original-input evidence, implementation identity |
| `run-record.yaml` | resolved protocol, stage records, reconstructed replay command |
| `forcefield.json` | force-field construction rebuilt only from recorded values |
| `file-inventory.yaml` | every source file with size, mtime and SHA-256 |
| `SHA256SUMS` | source tree plus this sidecar, excluding itself |
| `validation.json` | classification, evidence counts, what was not done |
""", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
