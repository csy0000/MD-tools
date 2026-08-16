"""`escort-explicit` — the installed entry point for portable explicit-solvent REST2.

Every subcommand works from any current directory once the wheel is installed; nothing here
resolves a path relative to the source checkout, and nothing reads
`docs/implementation/.../scripts`. Manifests may be given by path, or by the name of one shipped
inside the package (`--system cyclo_rgdfv`), which is what makes the documented commands free of
machine-local paths.

    escort-explicit validate-env
    escort-explicit validate-system  --system system.yaml
    escort-explicit prepare          --system system.yaml --experiment experiment.yaml \
                                     --out-root RUN_ROOT
    escort-explicit validate-bundle  --bundle BUNDLE_DIR
    escort-explicit rest2            --bundle BUNDLE_DIR --experiment experiment.yaml \
                                     --out-root RUN_ROOT --platform CUDA --device 0
    escort-explicit smoke            --system system.yaml --out-root RUN_ROOT --platform CPU
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
    system = load_system(_resolve_manifest(args.system, "system"))
    experiment = load_experiment(_resolve_manifest(args.experiment, "experiment"))
    envcheck.require_ok(platform=args.platform, device=args.device,
                        precision=experiment.doc["platform"]["precision"], route=system.route)
    runner.configure_device(args.platform, args.device)
    out = bundle_mod.prepare(system, experiment, Path(args.out_root),
                             platform=args.platform, device=args.device, name=args.name)
    print(f"bundle: {out}")
    print(bundle_mod.summarise(json.loads((out / "bundle_manifest.json").read_text())))
    return runner.EXIT_OK


def cmd_rest2(args) -> int:
    runner.install_signal_handlers()
    exp = _resolve_manifest(args.experiment, "experiment") if args.experiment else None
    code, run_dir = runner.launch_rest2(
        Path(args.bundle), exp, Path(args.out_root),
        platform=args.platform, device=args.device,
    )
    if run_dir is not None:
        status = runner.read_status(run_dir)
        print(f"run: {run_dir}")
        print(f"status: {status.get('status')}  "
              f"exchange rounds {status.get('observed_exchange_rounds')}"
              f"/{status.get('planned_exchange_rounds')}")
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
                                    platform=args.platform, device=args.device)
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

def build_parser() -> argparse.ArgumentParser:
    shipped = list_shipped()
    p = argparse.ArgumentParser(
        prog="escort-explicit",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version",
                   version=f"escort-ais {provenance.package_version()}")
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

    pr = sub.add_parser("prepare", help="build a portable bundle from manifests")
    pr.add_argument("--system", required=True)
    pr.add_argument("--experiment", required=True,
                    help=f"path, or a shipped manifest: {', '.join(shipped['experiments'])}")
    pr.add_argument("--out-root", required=True, metavar="RUN_ROOT")
    pr.add_argument("--name", default=None, help="bundle directory name (default: timestamped)")
    add_platform(pr)
    pr.set_defaults(func=cmd_prepare)

    r = sub.add_parser("rest2", help="run REST2 from a prepared bundle")
    r.add_argument("--bundle", required=True, metavar="BUNDLE_DIR")
    r.add_argument("--experiment", default=None,
                   help="override the experiment the bundle was prepared with")
    r.add_argument("--out-root", required=True, metavar="RUN_ROOT")
    add_platform(r)
    r.set_defaults(func=cmd_rest2)

    s = sub.add_parser("smoke", help="prepare + run the tiny shipped experiment end to end")
    s.add_argument("--system", required=True)
    s.add_argument("--experiment", default="smoke")
    s.add_argument("--out-root", required=True, metavar="RUN_ROOT")
    add_platform(s, default="CPU")
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
