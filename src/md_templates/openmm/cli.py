"""`md-openmm` — the installed entry point for portable explicit-solvent REST2.

Every subcommand works from any current directory once the wheel is installed; nothing here
resolves a path relative to the source checkout, and nothing reads
`docs/implementation/.../scripts`. Manifests may be given by path, or by the name of one shipped
inside the package (`--system cyclo_rgdfv`), which is what makes the documented commands free of
machine-local paths.

    md-openmm validate-env
    md-openmm validate-system  --system system.yaml
    md-openmm prepare          --system system.yaml --experiment experiment.yaml \
                                     --out-root RUN_ROOT
    md-openmm validate-bundle  --bundle BUNDLE_DIR
    md-openmm rest2            --bundle BUNDLE_DIR --experiment experiment.yaml \
                                     --out-root RUN_ROOT --platform CUDA --device 0
    md-openmm smoke            --system system.yaml --out-root RUN_ROOT --platform CPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import bundle as bundle_mod
from . import envcheck, provenance, runner
from .schemas import (
    ManifestError,
    PLATFORMS,
    list_shipped,
    load_experiment,
    load_system,
    shipped_experiment,
    shipped_system,
)


def _resolve_manifest(value: str, kind: str) -> Path:
    """Accept either a filesystem path or the name of a manifest shipped in the package."""
    path = Path(value)
    if path.exists():
        return path.resolve()
    shipped = shipped_system(value) if kind == "system" else shipped_experiment(value)
    if shipped.is_file():
        return shipped
    available = list_shipped()["systems" if kind == "system" else "experiments"]
    raise ManifestError(
        f"no such {kind} manifest: {value!r}\n"
        f"  give a path, or one of the shipped {kind} manifests: {', '.join(available)}"
    )


# ---------------------------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------------------------

def cmd_validate_env(args) -> int:
    checks = envcheck.run_checks(
        platform=args.platform, device=args.device, precision=args.precision, route=args.route
    )
    if args.json:
        print(json.dumps(envcheck.as_dict(checks), indent=2))
    else:
        print(envcheck.report(checks))
        bad = envcheck.errors(checks)
        print()
        print(f"{len(bad)} blocking problem(s)." if bad else "environment OK.")
    return runner.EXIT_ENVIRONMENT if envcheck.errors(checks) else runner.EXIT_OK


def cmd_validate_system(args) -> int:
    path = _resolve_manifest(args.system, "system")
    system = load_system(path, check_chemistry=not args.no_chemistry)
    print(f"valid: {path}")
    print(f"  system_id     {system.system_id}   ({system.display_name})")
    print(f"  route         {system.route}")
    print(f"  molecule      {system.canonical_hash}")
    if system.route == "smiles":
        print(f"  canonical     {system.doc['input']['canonical_isomeric_smiles']}")
        print(f"  formal charge {system.formal_charge}")
    par = system.doc["parameterization"]
    print(f"  forcefields   small_molecule={par['small_molecule_forcefield']} "
          f"protein={par['protein_forcefield']} water={par['water_forcefield']}")
    solv = system.doc.get("solvation") or {}
    if solv:
        # box shape is part of the identity of the prepared system, not a cosmetic default: the
        # ladder evidence was measured in one geometry and is not transferable to another
        print(f"  solvation     box_shape={solv.get('box_shape')} "
              f"padding_nm={solv.get('padding_nm')} "
              f"ionic_strength_molar={solv.get('ionic_strength_molar')} "
              f"water_model={solv.get('water_model')}")
    if args.experiment:
        experiment = load_experiment(_resolve_manifest(args.experiment, "experiment"))
        print(f"  experiment    {experiment.experiment_id}   {experiment.n_rungs} rungs   "
              f"ladder_status={experiment.ladder_status}")
        if experiment.ladder_status != "pilot_supported":
            print("                (no system-specific pilot evidence stands behind this ladder)")
        else:
            print("                (system-specific pilot evidence; NOT convergence or "
                  "production readiness)")
    return runner.EXIT_OK


def cmd_validate_bundle(args) -> int:
    manifest = bundle_mod.validate_bundle(Path(args.bundle))
    print(f"valid bundle: {Path(args.bundle).resolve()}")
    print(bundle_mod.summarise(manifest))
    return runner.EXIT_OK


def cmd_prepare(args) -> int:
    if getattr(args, "config", None):
        return _prepare_from_canonical(args)
    if not args.system or not args.experiment:
        print("prepare needs either --config CANONICAL_DOCUMENT, or both --system and "
              "--experiment (the legacy manifest pair)", file=sys.stderr)
        return runner.EXIT_MANIFEST
    system = load_system(_resolve_manifest(args.system, "system"))
    experiment = load_experiment(_resolve_manifest(args.experiment, "experiment"))
    envcheck.require_ok(platform=args.platform, device=args.device,
                        precision=experiment.doc["platform"]["precision"], route=system.route)
    runner.configure_device(args.platform, args.device)
    out = bundle_mod.prepare(system, experiment, Path(args.out_root),
                             platform=args.platform, device=args.device, name=args.name,
                             omega_exclusion=args.omega_exclusion)
    print(f"bundle: {out}")
    print(bundle_mod.summarise(json.loads((out / "bundle_manifest.json").read_text())))
    return runner.EXIT_OK


def cmd_rest2(args) -> int:
    runner.install_signal_handlers()
    exp = _resolve_manifest(args.experiment, "experiment") if args.experiment else None
    code, run_dir = runner.launch_rest2(
        Path(args.bundle), exp, Path(args.out_root),
        platform=args.platform, device=args.device, omega_exclusion=args.omega_exclusion,
        run_name=args.run_name, resume_run=_resolve_resume(args),
    )
    if run_dir is not None:
        status = runner.read_status(run_dir)
        print(f"run: {run_dir}")
        print(f"status: {status.get('status')}  "
              f"exchange rounds {status.get('observed_exchange_rounds')}"
              f"/{status.get('planned_exchange_rounds')}")
    return code



def cmd_md(args) -> int:
    """Conventional explicit-water MD: one walker, no replicas, no exchange."""
    runner.install_signal_handlers()
    exp = _resolve_manifest(args.experiment, "experiment") if args.experiment else None
    code, run_dir = runner.launch_md(
        Path(args.bundle), exp, Path(args.out_root),
        platform=args.platform, device=args.device,
        run_name=args.run_name, resume_run=_resolve_resume(args),
    )
    if run_dir is not None:
        status = runner.read_status(run_dir)
        print(f"run: {run_dir}")
        print(f"status: {status.get('status')}  chunks {status.get('lifetime_chunks')}"
              f"/{status.get('planned_chunks')} (lifetime/this invocation's target)")
    return code


def cmd_smoke(args) -> int:
    """prepare + rest2 with the tiny shipped experiment, as one command.

    This is the command an external repository runs first: it proves the package is installed,
    the toolchain is complete, and a System can be built, propagated and exchanged on this
    machine. It proves nothing scientific.
    """
    system = load_system(_resolve_manifest(args.system, "system"))
    experiment = load_experiment(_resolve_manifest(args.experiment, "experiment"))
    envcheck.require_ok(platform=args.platform, device=args.device,
                        precision=experiment.doc["platform"]["precision"], route=system.route)
    runner.install_signal_handlers()
    runner.configure_device(args.platform, args.device)

    out_root = Path(args.out_root)
    print(f"[smoke] preparing {system.system_id} ({experiment.experiment_id}) ...")
    bundle_dir = bundle_mod.prepare(system, experiment, out_root,
                                    platform=args.platform, device=args.device,
                                    omega_exclusion=args.omega_exclusion)
    print(f"[smoke] bundle: {bundle_dir}")
    print("[smoke] running REST2 ...")
    code, run_dir = runner.launch_rest2(bundle_dir, experiment.source, out_root,
                                        platform=args.platform, device=args.device)
    status = runner.read_status(run_dir) if run_dir else {}
    print(f"[smoke] run: {run_dir}")
    print(f"[smoke] status: {status.get('status')}  "
          f"exchange rounds {status.get('observed_exchange_rounds')}"
          f"/{status.get('planned_exchange_rounds')}")
    if status.get("status") != runner.STATUS_COMPLETED:
        print("[smoke] FAILED: the smoke did not complete its (tiny) budget", file=sys.stderr)
        return code or runner.EXIT_RUNTIME
    if int(status.get("observed_exchange_rounds") or 0) < 2:
        print("[smoke] FAILED: fewer than two exchange rounds were recorded", file=sys.stderr)
        return runner.EXIT_RUNTIME
    print("[smoke] OK — installable and mechanically executable on this machine.")
    print("[smoke] This says nothing about a macrocycle ladder; see PORTABLE_REST2.md.")
    return runner.EXIT_OK


# ---------------------------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------------------------

def _bool_flag(text: str) -> bool:
    """Parse `true|false` for a setting that changes the Hamiltonian.

    Deliberately strict. `--omega-exclusion maybe` must not fall through to a default: the two
    values are different physics, so an unparsed one is an error, not a preference.
    """
    lowered = str(text).strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"expected true or false, got {text!r}; this setting selects which torsions REST2 scales, "
        "so it is not guessed"
    )


def add_omega_exclusion(sp) -> None:
    """`--omega-exclusion true|false`, plus `--no-omega-exclusion` onto the same setting."""
    group = sp.add_mutually_exclusive_group()
    group.add_argument("--omega-exclusion", dest="omega_exclusion", type=_bool_flag,
                       default=None, metavar="true|false",
                       help="exclude ordinary amide omega torsions from REST2 scaling "
                            "(default: true, from rest2.omega_exclusion)")
    group.add_argument("--no-omega-exclusion", dest="omega_exclusion",
                       action="store_false",
                       help="equivalent to --omega-exclusion false")



def add_run_naming(sp) -> None:
    """`--run-name` for a fresh run, `--resume-run` to continue one, never both."""
    group = sp.add_mutually_exclusive_group()
    group.add_argument("--run-name", default=None, metavar="NAME",
                       help="directory name for a FRESH run: creates exactly <out-root>/<NAME>, "
                            "with no timestamp or hash appended. Omit for a timestamped default.")
    group.add_argument("--resume-run", default=None, metavar="PATH_OR_NAME",
                       help="continue an existing run IN PLACE. Not a new directory, not a "
                            "sibling: the same run, extended by this invocation's n_chunks.")


def _resolve_resume(args) -> Optional[Path]:
    """`--resume-run` may be a path or a bare name under --out-root."""
    if getattr(args, "resume_run", None) is None:
        return None
    candidate = Path(args.resume_run)
    if candidate.is_dir():
        return candidate
    return Path(args.out_root) / args.resume_run



# ---------------------------------------------------------------------------------------------
# `config` -- the canonical configuration front end
# ---------------------------------------------------------------------------------------------

def _spec_modules():
    from .spec import canonical, diffs, migrate, resolve
    return canonical, diffs, migrate, resolve


def _emit(payload, args) -> None:
    """Write JSON or YAML to --output or stdout, so every config command composes with a pipe."""
    canonical, _, _, _ = _spec_modules()
    fmt = getattr(args, "format", "json") or "json"
    if fmt == "yaml":
        import yaml
        text = yaml.safe_dump(canonical.to_plain(payload), sort_keys=False)
    else:
        text = json.dumps(canonical.to_plain(payload), indent=2, sort_keys=False) + "\n"
    out = getattr(args, "output", None)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"written: {out}")
    else:
        sys.stdout.write(text)


def cmd_config_list_profiles(args) -> int:
    canonical, _, _, resolve = _spec_modules()
    rows = []
    for doc in resolve.list_profiles():
        rows.append({"profile_id": doc["profile_id"],
                     "profile_schema_version": doc["profile_schema_version"],
                     "route": doc.get("route"), "method": doc.get("method"),
                     "description": doc.get("description"),
                     "sha256": canonical.sha256_of({k: v for k, v in doc.items()
                                                    if k != "_path"})})
    _emit(rows, args)
    return runner.EXIT_OK


def cmd_config_validate(args) -> int:
    """Full schema and cross-field validation, without constructing anything in OpenMM."""
    _, _, _, resolve = _spec_modules()
    document = resolve.load_document(Path(args.input))
    result = resolve.resolve_spec(document, overrides=args.set, profile_id=args.profile)
    print(f"valid: {args.input}")
    print(f"  profile  {result['profile']['profile_id']} "
          f"(v{result['profile']['profile_schema_version']})")
    print(f"  method   {result['spec'].method}   route {result['spec'].system.route}")
    for name, value in result["hashes"].items():
        print(f"  {name:22} {value}")
    return runner.EXIT_OK


def cmd_config_resolve(args) -> int:
    canonical, _, _, resolve = _spec_modules()
    document = resolve.load_document(Path(args.input))
    result = resolve.resolve_spec(document, overrides=args.set, profile_id=args.profile)
    _emit({"profile": result["profile"],
           "hashes": result["hashes"],
           "sources": result["sources"],
           "derived": {"protocol.production.total_ps": result["spec"].protocol.production.total},
           "configuration": canonical.dump_model(result["spec"])}, args)
    return runner.EXIT_OK


def cmd_config_diff(args) -> int:
    _, diffs, _, resolve = _spec_modules()
    a = resolve.resolve_spec(resolve.load_document(Path(args.input_a)))["spec"]
    b = resolve.resolve_spec(resolve.load_document(Path(args.input_b)))["spec"]
    rows = diffs.diff_specs(a, b)
    if not rows:
        print("identical: the two documents resolve to the same canonical configuration")
        return runner.EXIT_OK
    width = max(len(r["field"]) for r in rows)
    for row in rows:
        print(f"  {row['consequence']:<20} {row['field']:<{width}}  {row['a']!r} -> {row['b']!r}")
    consequences = sorted({r["consequence"] for r in rows})
    print(f"\n{len(rows)} difference(s); consequences: {', '.join(consequences)}")
    return runner.EXIT_OK


def cmd_config_explain(args) -> int:
    _, diffs, _, resolve = _spec_modules()
    result = resolve.resolve_spec(resolve.load_document(Path(args.input)))
    try:
        info = diffs.explain_field(result["spec"], args.field, result["sources"])
    except KeyError as exc:
        print(f"{exc}", file=sys.stderr)
        return runner.EXIT_MANIFEST
    _emit(info, args)
    return runner.EXIT_OK


def cmd_config_migrate(args) -> int:
    _, _, migrate, resolve = _spec_modules()
    from .schemas import load_experiment, load_system

    system = load_system(_resolve_manifest(args.input, "system"), check_chemistry=False)
    experiment = load_experiment(_resolve_manifest(args.experiment, "experiment"))
    document, notes = migrate.migrate_manifests(system.doc, experiment.doc)
    print("semantic changes:")
    for note in notes:
        print(f"  - {note}")
    if args.output:
        target = Path(args.output)
        if target.exists() and not args.overwrite:
            print(f"refusing to overwrite {target} (pass --overwrite)", file=sys.stderr)
            return runner.EXIT_MANIFEST
        target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"written: {target}")
    else:
        sys.stdout.write(json.dumps(document, indent=2) + "\n")
    return runner.EXIT_OK


def cmd_config_init(args) -> int:
    _, _, _, resolve = _spec_modules()
    profile = (resolve.load_profile(args.profile) if args.profile
               else resolve.select_profile(args.route, args.method))
    example_system = ({"system_id": "my_ligand", "route": "smiles", "smiles": "CCO"}
                      if args.route == "smiles"
                      else {"system_id": "my_peptide", "route": "pdb", "pdb": "input.pdb"})
    document = {
        "profile": profile["profile_id"],
        "system": example_system,
        "protocol": {"production": {"method": args.method, "n_chunks": 2, "chunk": "10 ps"}},
        "execution": {"platform": "CPU"},
    }
    header = (f"# Generated by `md-openmm config init`.\n"
              f"# Defaults come from profile {profile['profile_id']} "
              f"(v{profile['profile_schema_version']}).\n"
              f"# Only the fields you want to override need to appear here.\n"
              f"# Validate with:  md-openmm config validate THIS_FILE\n")
    target = Path(args.output)
    if target.exists() and not args.overwrite:
        print(f"refusing to overwrite {target} (pass --overwrite)", file=sys.stderr)
        return runner.EXIT_MANIFEST
    if target.suffix.lower() in (".yaml", ".yml"):
        import yaml
        target.write_text(header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    else:
        target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"written: {target}   (profile {profile['profile_id']})")
    return runner.EXIT_OK



def _prepare_from_canonical(args) -> int:
    """Prepare a bundle from a canonical configuration document.

    The canonical model is the source of truth; the manifest path reaches the same builder by being
    migrated into this model first. There is one configuration engine, not two.
    """
    canonical, _, _, resolve = _spec_modules()
    from .spec.adapter import spec_to_runtime_cfg

    result = resolve.resolve_spec(resolve.load_document(Path(args.config)),
                                  overrides=getattr(args, "set", None),
                                  profile_id=getattr(args, "profile", None))
    spec = result["spec"]
    cfg = spec_to_runtime_cfg(spec)
    if spec.system.route != "smiles":
        print("prepare from a canonical document currently supports the smiles route only; "
              "the pdb route still uses --system/--experiment", file=sys.stderr)
        return runner.EXIT_MANIFEST

    system = _synthetic_system_manifest(spec)
    experiment = load_experiment(_resolve_manifest("smoke", "experiment"))
    envcheck.require_ok(platform=args.platform, device=args.device,
                        precision=spec.execution.precision, route=spec.system.route)
    runner.configure_device(args.platform, args.device)
    out = bundle_mod.prepare(
        system, experiment, Path(args.out_root), platform=args.platform, device=args.device,
        name=args.name, resolved_cfg=cfg,
        # the EXACT document the user supplied travels into the bundle
        original_config=Path(args.config),
        original_input=(Path(args.config).parent / spec.system.pdb
                        if spec.system.route == "pdb" and spec.system.pdb else None),
        canonical={"profile": result["profile"], "hashes": result["hashes"],
                   "sources": result["sources"],
                   # to_plain, not dump_model: the stored record must be in CANONICAL form
                   # (value + unit), not a bare tuple whose meaning depends on field order.
                   "configuration": canonical.to_plain(canonical.dump_model(spec))},
    )
    print(f"bundle: {out}")
    print(f"  profile {result['profile']['profile_id']}  "
          f"system_build {result['hashes']['system_build_sha256'][:12]}")
    return runner.EXIT_OK


def _synthetic_system_manifest(spec):
    """A SystemManifest view of the canonical system section, for the existing builder.

    The builder still addresses molecular identity through a manifest object; this adapts the
    canonical model onto that interface rather than duplicating the builder.
    """
    import tempfile

    import yaml

    from .schemas import load_system

    doc = {
        "schema_version": 1,
        "system_id": spec.system.system_id,
        "display_name": spec.system.display_name or spec.system.system_id,
        "input": {"route": "smiles", "smiles": spec.system.smiles,
                  "expected_formal_charge": spec.system.expected_formal_charge},
        "parameterization": {
            "small_molecule_forcefield": spec.build.forcefield.small_molecule,
            "charge_method": spec.build.forcefield.charge_method,
            "protein_forcefield": spec.build.forcefield.protein,
            "water_forcefield": spec.build.forcefield.water,
        },
        "solvation": {
            "water_model": spec.build.solvation.water_model,
            "box_shape": spec.build.solvation.box_shape,
            "padding_nm": spec.build.solvation.padding.value,
            "ionic_strength_molar": spec.build.solvation.ionic_strength_molar,
            "positive_ion": spec.build.solvation.positive_ion,
            "negative_ion": spec.build.solvation.negative_ion,
        },
    }
    # The canonical model treats the canonical SMILES and its hash as OPTIONAL, because they are
    # derived from the declared SMILES rather than independently chosen. Derive them here when the
    # document did not state them, and check them when it did -- a stated value that disagrees with
    # RDKit means the document names one molecule and describes another.
    from .schemas import sha256_text

    canonical_smiles = spec.system.canonical_isomeric_smiles
    if canonical_smiles is None:
        try:
            from rdkit import Chem
        except ImportError:                                    # pragma: no cover
            raise SystemExit(
                "system.canonical_isomeric_smiles is absent and RDKit is unavailable to derive it; "
                "state it explicitly in the document."
            ) from None
        mol = Chem.MolFromSmiles(spec.system.smiles)
        if mol is None:
            raise SystemExit(f"system.smiles {spec.system.smiles!r} is not parseable by RDKit")
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    doc["input"]["canonical_isomeric_smiles"] = canonical_smiles
    doc["input"]["canonical_smiles_sha256"] = (spec.system.canonical_smiles_sha256
                                               or sha256_text(canonical_smiles))
    tmp = Path(tempfile.mkdtemp()) / f"{spec.system.system_id}.yaml"
    tmp.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return load_system(tmp, check_chemistry=False)



# ---------------------------------------------------------------------------------------------
# `bundle` -- validate, inspect, relocate-check
# ---------------------------------------------------------------------------------------------

def cmd_bundle_validate(args) -> int:
    from . import bundlecheck

    report = bundlecheck.validate_bundle_v2(Path(args.bundle), deep=args.deep)
    print(f"{'valid' if report.ok else 'INVALID'}: {report['bundle']}")
    print(f"  contract: {report.get('contract')}")
    for warning in report.get("warnings") or []:
        print(f"  [warn]  {warning}")
    for error in report.get("errors") or []:
        print(f"  [error] {error}")
    return runner.EXIT_OK if report.ok else runner.EXIT_BUNDLE


def cmd_bundle_inspect(args) -> int:
    from . import bundlecheck

    info = bundlecheck.inspect_bundle(Path(args.bundle))
    if args.format in ("json", "yaml"):
        _emit(info, args)
        return runner.EXIT_OK
    counts = info["counts"]
    print(f"bundle {info['bundle_id']}   schema v{info['bundle_schema_version']}")
    print(f"  system      {info['identity']['system_id']}  route {info['identity']['route']}")
    print(f"  methods     {', '.join(info['methods_supported'])}")
    print(f"  profile     {info['profile'].get('profile_id')} "
          f"v{info['profile'].get('profile_schema_version')}")
    ff = info["forcefields"]
    print(f"  forcefield  small_molecule={ff['small_molecule']} protein={ff['protein']} "
          f"water={ff['water']}")
    print(f"  counts      topology_atoms={counts.get('topology_atoms')} "
          f"openmm_particles={counts.get('openmm_particles')} "
          f"virtual_sites={counts.get('virtual_sites')} "
          f"massless={counts.get('massless_particles')} "
          f"constraints={counts.get('constraints')} dof={counts.get('degrees_of_freedom')}")
    for name, value in (info["hashes"] or {}).items():
        print(f"  {name:26} {value[:16]}")
    print(f"  validation  {'ok' if info['validation']['ok'] else 'FAILED'} "
          f"({info['validation']['contract']})")
    for warning in info["validation"]["warnings"] or []:
        print(f"  [warn]      {warning}")
    for error in info["validation"]["errors"] or []:
        print(f"  [error]     {error}")
    return runner.EXIT_OK if info["validation"]["ok"] else runner.EXIT_BUNDLE


def cmd_bundle_relocate_check(args) -> int:
    from . import bundlecheck

    result = bundlecheck.relocate_check(Path(args.bundle))
    print(f"{result['verdict']}: {result['source']}")
    print(f"  validated after copying to an unrelated directory: {result['relocated_ok']}")
    for item in result["references_escaping_the_bundle"]:
        print(f"  [escape] {item}")
    for error in result["errors_after_relocation"] or []:
        print(f"  [error]  {error}")
    return runner.EXIT_OK if result["verdict"] == "relocatable" else runner.EXIT_BUNDLE


def build_parser() -> argparse.ArgumentParser:
    shipped = list_shipped()
    p = argparse.ArgumentParser(
        prog="md-openmm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version",
                   version=f"md-templates {provenance.package_version()}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_platform(sp, default: str = "CUDA") -> None:
        sp.add_argument("--platform", choices=PLATFORMS, default=default,
                        help=f"OpenMM platform (default: {default})")
        sp.add_argument("--device", default=None, metavar="DEVICE_ID",
                        help="device index; only meaningful for CUDA/OpenCL. "
                             "CUDA_DEVICE_ORDER=PCI_BUS_ID is set so this matches nvidia-smi.")

    e = sub.add_parser("validate-env", help="check this machine can prepare and run")
    e.add_argument("--platform", choices=PLATFORMS, default=None)
    e.add_argument("--device", default=None, metavar="DEVICE_ID")
    e.add_argument("--precision", default="mixed", choices=("single", "mixed", "double"))
    e.add_argument("--route", default="smiles", choices=("smiles", "pdb"),
                   help="which input route to check the optional dependencies for")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_validate_env)

    v = sub.add_parser("validate-system", help="validate a system manifest")
    v.add_argument("--system", required=True,
                   help=f"path, or a shipped manifest: {', '.join(shipped['systems'])}")
    v.add_argument("--experiment", default=None,
                   help="also validate this experiment and report its ladder status")
    v.add_argument("--no-chemistry", action="store_true",
                   help="skip the RDKit checks (structure only); never used by prepare/rest2")
    v.set_defaults(func=cmd_validate_system)

    b = sub.add_parser("validate-bundle", help="recompute a bundle's hashes")
    b.add_argument("--bundle", required=True, metavar="BUNDLE_DIR")
    b.set_defaults(func=cmd_validate_bundle)

    pr = sub.add_parser("prepare", help="build a portable bundle")
    pr.add_argument("--config", default=None,
                    help="canonical configuration document (.yaml/.yml/.json). Preferred; the "
                         "--system/--experiment manifest pair is the legacy front end and is "
                         "migrated into the same canonical model by `config migrate`.")
    pr.add_argument("--profile", default=None, help="pin a profile when using --config")
    pr.add_argument("--set", action="append", default=None, metavar="dotted.path=value")
    pr.add_argument("--system", required=False)
    pr.add_argument("--experiment", required=False,
                    help=f"path, or a shipped manifest: {', '.join(shipped['experiments'])}")
    pr.add_argument("--out-root", required=True, metavar="RUN_ROOT")
    pr.add_argument("--name", default=None, help="bundle directory name (default: timestamped)")
    add_platform(pr)
    add_omega_exclusion(pr)
    pr.set_defaults(func=cmd_prepare)

    r = sub.add_parser("rest2", help="run REST2 from a prepared bundle")
    r.add_argument("--bundle", required=True, metavar="BUNDLE_DIR")
    r.add_argument("--experiment", default=None,
                   help="override the experiment the bundle was prepared with")
    r.add_argument("--out-root", required=True, metavar="RUN_ROOT")
    add_platform(r)
    add_omega_exclusion(r)
    add_run_naming(r)
    r.set_defaults(func=cmd_rest2)

    c = sub.add_parser("config", help="inspect and resolve simulation configuration")
    csub = c.add_subparsers(dest="config_command", required=True)

    lp = csub.add_parser("list-profiles", help="the packaged versioned default profiles")
    lp.add_argument("--format", choices=("json", "yaml"), default="json")
    lp.add_argument("--output", default=None)
    lp.set_defaults(func=cmd_config_list_profiles)

    ci = csub.add_parser("init", help="write a runnable template document")
    ci.add_argument("--method", required=True, choices=("md", "rest2"))
    ci.add_argument("--route", required=True, choices=("pdb", "smiles"))
    ci.add_argument("--profile", default=None)
    ci.add_argument("--output", required=True)
    ci.add_argument("--overwrite", action="store_true")
    ci.set_defaults(func=cmd_config_init)

    cv = csub.add_parser("validate", help="schema and cross-field validation, no OpenMM")
    cv.add_argument("input")
    cv.add_argument("--profile", default=None)
    cv.add_argument("--set", action="append", default=None, metavar="dotted.path=value")
    cv.set_defaults(func=cmd_config_validate)

    cr = csub.add_parser("resolve", help="the fully expanded configuration, sources and hashes")
    cr.add_argument("input")
    cr.add_argument("--profile", default=None)
    cr.add_argument("--set", action="append", default=None, metavar="dotted.path=value")
    cr.add_argument("--format", choices=("json", "yaml"), default="json")
    cr.add_argument("--output", default=None)
    cr.set_defaults(func=cmd_config_resolve)

    cd = csub.add_parser("diff", help="classify differences by their consequence")
    cd.add_argument("input_a")
    cd.add_argument("input_b")
    cd.set_defaults(func=cmd_config_diff)

    ce = csub.add_parser("explain", help="type, unit, source, allowed values, scientific effect")
    ce.add_argument("input")
    ce.add_argument("field", metavar="DOTTED.FIELD")
    ce.add_argument("--format", choices=("json", "yaml"), default="json")
    ce.add_argument("--output", default=None)
    ce.set_defaults(func=cmd_config_explain)

    cm = csub.add_parser("migrate", help="convert shipped system+experiment manifests")
    cm.add_argument("input", help="system manifest (path or shipped name)")
    cm.add_argument("--experiment", required=True)
    cm.add_argument("--output", default=None)
    cm.add_argument("--overwrite", action="store_true")
    cm.set_defaults(func=cmd_config_migrate)

    bl = sub.add_parser("bundle", help="validate, inspect and relocation-check a bundle")
    blsub = bl.add_subparsers(dest="bundle_command", required=True)

    bv = blsub.add_parser("validate", help="schema, roles, paths, checksums, hashes, counts")
    bv.add_argument("bundle", metavar="BUNDLE")
    bv.add_argument("--deep", action="store_true",
                    help="also deserialize the System/State and cross-check; still runs no dynamics")
    bv.set_defaults(func=cmd_bundle_validate)

    bi = blsub.add_parser("inspect", help="summarize a bundle without constructing a Context")
    bi.add_argument("bundle", metavar="BUNDLE")
    bi.add_argument("--format", choices=("text", "json", "yaml"), default="text")
    bi.add_argument("--output", default=None)
    bi.set_defaults(func=cmd_bundle_inspect)

    br = blsub.add_parser("relocate-check",
                          help="copy elsewhere, validate there, report anything that escaped")
    br.add_argument("bundle", metavar="BUNDLE")
    br.set_defaults(func=cmd_bundle_relocate_check)

    m = sub.add_parser("md", help="conventional explicit-water MD from a prepared bundle")
    m.add_argument("--bundle", required=True, metavar="BUNDLE_DIR")
    m.add_argument("--experiment", default=None,
                   help="override the experiment the bundle was prepared with")
    m.add_argument("--out-root", required=True, metavar="RUN_ROOT")
    add_platform(m)
    add_run_naming(m)
    m.set_defaults(func=cmd_md)

    s = sub.add_parser("smoke", help="prepare + run the tiny shipped experiment end to end")
    s.add_argument("--system", required=True)
    s.add_argument("--experiment", default="smoke")
    s.add_argument("--out-root", required=True, metavar="RUN_ROOT")
    add_platform(s, default="CPU")
    add_omega_exclusion(s)
    s.set_defaults(func=cmd_smoke)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return runner.EXIT_MANIFEST
    except bundle_mod.BundleError as exc:
        print(f"bundle error: {exc}", file=sys.stderr)
        return runner.EXIT_BUNDLE
    except runner.IncompatibleExperiment as exc:
        print(f"incompatible experiment: {exc}", file=sys.stderr)
        return runner.EXIT_INCOMPATIBLE
    except runner.RunExists as exc:
        print(f"{exc}", file=sys.stderr)
        return runner.EXIT_RUN_EXISTS
    except EnvironmentError as exc:
        print(f"environment error: {exc}", file=sys.stderr)
        return runner.EXIT_ENVIRONMENT
    except ValueError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return runner.EXIT_USAGE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return runner.EXIT_INTERRUPTED


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
