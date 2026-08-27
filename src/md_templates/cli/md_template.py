"""`md-template` -- set up an MD installation directory and put OpenMM in it.

Two subcommands, both operating on one directory named by `$MD_STACK` or `--target-dir`. There is
no hidden state anywhere else: everything this command knows lives in `$MD_STACK/machine.yaml`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..install.inspect import initialise, load_machine
from ..install.openmm import InstallError, install_openmm, validate_existing


def _stack_from(args) -> Path:
    """`--target-dir`, else `$MD_STACK`, else an error that says how to fix it."""
    if getattr(args, "target_dir", None):
        return Path(args.target_dir).expanduser().resolve()
    env = os.environ.get("MD_STACK")
    if env:
        return Path(env).expanduser().resolve()
    raise SystemExit(
        "no MD stack directory. Pass --target-dir, or set the environment variable:\n"
        "    export MD_STACK=/absolute/path/to/MD_STACK")


def cmd_init(args) -> int:
    stack = _stack_from(args)
    path = initialise(stack)
    machine = load_machine(stack)
    gpu = machine.get("gpu") or {}
    hardware = machine.get("machine") or {}

    print(f"  initialised   : {stack}")
    print(f"  machine.yaml  : {path}")
    print(f"  cpu           : {hardware.get('physical_cpu_cores')} physical / "
          f"{hardware.get('logical_cpu_cores')} logical cores, "
          f"{hardware.get('memory_gb')} GB")
    if gpu.get("available"):
        print(f"  gpu           : {len(gpu.get('devices') or [])} device(s), "
              f"driver {gpu.get('driver_version')}, CUDA {gpu.get('cuda_version')}")
        for device in gpu.get("devices") or []:
            print(f"                  [{device['index']}] {device['name']}  "
                  f"{device['memory_gb']} GB  {device['uuid']}")
    else:
        print("  gpu           : none detected (CPU platforms only)")
    print()
    print("Set this so later commands find the stack:")
    print(f"    export MD_STACK={stack}")
    return 0


def _report_environment(result: dict) -> None:
    """What the environment can actually do, not what was requested."""
    versions = result.get("versions") or {}
    executables = result.get("executables") or {}
    print(f"  openmm        : {result.get('openmm_version')} "
          f"(python {result.get('python_version')})")
    release = result.get("release") or {}
    package = release.get("package") or {}
    if package.get("version"):
        print(f"  openmm package: {package['version']} build {package.get('build')} "
              f"from {package.get('channel')}  -> {release.get('build_kind')}")
    if release.get("runtime_string_carries_build_marker"):
        # The ordinary conda-forge case, and confusing enough to name every time: the RELEASE
        # package reports a version string that looks like a development build.
        print(f"  version note  : OpenMM reports {release.get('runtime_version')!r} at runtime "
              f"(short_version {release.get('runtime_short_version')!r}); the release identity "
              f"comes from {release.get('authority')}")
    if release.get("exact_release_required"):
        verdict = "exact stable release" if release.get("is_exact_release") else "NOT the release"
        print(f"  release check : {release.get('requested_version')} -> {verdict}")
    for module in ("yaml", "numpy", "openff.toolkit", "openff.nagl_models",
                   "openmmforcefields", "parmed", "rdkit", "mdtraj"):
        if module in versions:
            print(f"  {module:<14}: {versions[module]}")
    print(f"  ambertools    : " + ", ".join(
        f"{name} {'ok' if executables.get(name) else 'MISSING'}"
        for name in ("sqm", "antechamber", "tleap")))
    print(f"  am1bcc        : {'ready' if result.get('am1bcc_ready') else 'NOT available'}")
    print(f"  platforms     : {', '.join(result.get('platforms') or [])}")
    print(f"  reference     : {result.get('reference_check')}")
    print(f"  cpu           : {result.get('cpu_check')}")
    print(f"  cuda          : {result.get('cuda_check')}")
    for note in result.get("warnings") or []:
        print(f"  note          : {note}")
    print()
    print("Recorded in machine.yaml under `installed.openmm`.")


def cmd_install(args) -> int:
    stack = _stack_from(args)
    engine = str(args.engine).lower()
    if engine != "openmm":
        raise SystemExit(
            f"engine {args.engine!r} is not supported. This repository installs OpenMM only; "
            "Amber and GROMACS are not implemented and are not presented as available.")

    if args.validate:
        # Validate an environment that already exists instead of building one. The check is
        # identical; only its subject differs.
        # `--expect-version` is what turns validation of a stack environment into a release
        # check. Without it a user validating an environment built around a different OpenMM gets
        # the facts and no refusal, which is what this flag exists to keep true.
        try:
            result = validate_existing(stack, args.validate,
                                       version=(args.expect_version or ""))
        except (InstallError, FileNotFoundError) as error:
            raise SystemExit(str(error))
        print(f"  environment   : {result['prefix']}  (validated, not created)")
        _report_environment(result)
        return 0

    try:
        result = install_openmm(stack, args.engine_version, dry_run=args.dry_run)
    except (InstallError, FileNotFoundError) as error:
        raise SystemExit(str(error))

    print(f"  environment   : {result['prefix']}")
    print(f"  log           : {result['log']}")
    if result.get("dry_run"):
        print("  dry run: nothing was installed")
        return 0
    _report_environment(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="md-template",
        description="Set up an MD installation directory and install OpenMM into it.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create the stack layout and inspect the machine")
    # both spellings: --target-dir is the documented one, --target_dir the original
    init.add_argument("--target-dir", "--target_dir", dest="target_dir", default=None,
                      help="stack directory (default: $MD_STACK)")
    init.set_defaults(func=cmd_init)

    install = sub.add_parser("install", help="install an engine into the stack")
    install.add_argument("-e", "--engine", default="openmm", help="only 'openmm' is supported")
    install.add_argument("-ev", "--engine-version", "--engine_version", dest="engine_version",
                         default="8.6.0")
    install.add_argument("--target-dir", "--target_dir", dest="target_dir", default=None)
    install.add_argument("--dry-run", action="store_true",
                         help="write the command that would run, install nothing")
    install.add_argument("--validate", metavar="PREFIX", default=None,
                         help="validate an environment that already exists instead of creating "
                              "one, and record the result in machine.yaml")
    install.add_argument("--expect-version", "--expect_version", dest="expect_version",
                         default=None, metavar="VERSION",
                         help="with --validate: require this exact OpenMM release. 8.6.0 is held "
                              "to the stable release, so a development build reporting the same "
                              "short_version (8.6.0.dev-*) is refused rather than accepted.")
    install.set_defaults(func=cmd_install)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":                         # pragma: no cover
    raise SystemExit(main())
