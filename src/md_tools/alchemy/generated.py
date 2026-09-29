"""The step between a resolved alchemical configuration and a running window (0.7.0, X1).

ONE implementation, two surfaces. `md-openmm md-run -i alchemical.in` and `python w003.py` both
reach `window_main`, which is the only place a window is dispatched. `md-run` is a surface: it
resolves and hands off, exactly as it does for `stage_main`, `replica_main` and `ais_main`. If the
window loop existed here and again in `md-run`, the two would eventually disagree about which
windows a run covers, and only one of them would be the one that ran.

WHAT A GENERATED DIRECTORY HOLDS, and why each piece is there

    alchemical-run1/
      resolved.config        the single resolved declaration; every script reads the copy beside
                             itself, located from __file__, so the directory is movable
      plan.<digest>/         THE PLAN, COPIED IN, content-addressed. A generated directory that
                             referred to a plan somewhere else would break the moment it was
                             copied to a cluster -- and break SILENTLY where the path exists and
                             holds a different plan. The digest in the name is why a plan that
                             changed cannot quietly replace the one this run was generated from
      leg/                   the leg `build-md` prepared: system.xml (the Hamiltonian's System,
                             built ONCE at generation), topology.pdb, leg.json, plan.json
      w000.py .. wNNN.py     the entry points, one per window, two lines each
      run.sh                 every window in order, through `md-openmm md-run`

THE HAMILTONIAN IS REBUILT AND THEN CHECKED, not trusted either way. The run rebuilds it from the
copied-in plan (`from_plan`) and `run_leg` compares the result against the `system_sha256` that
`prepare_leg` recorded when `build-md` wrote `leg/system.xml`. So a Hamiltonian that rebuilt
differently -- a changed softcore default, a different package resolving under the same name --
is refused by name rather than integrated. That is the same shape as a REST2 ladder checking its
saved states against `scaler.yaml`, with the digest doing the work the file identity does there.

A WINDOW IS NOT A STAGE. It has no `-c` chain, no restart handed forward, no ordering: windows are
independent and a campaign is complete when every one of them is. `--window` selects a subset the
way AIS's `--paths` does, and the same window run twice is skipped by its own completion record
rather than rerun.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

#: The window ids `build-md` writes, and the runtime reads. `w000` is the first window of the
#: path, in increasing lambda; the index is the WINDOW's, and the file it writes carries it.
WINDOW_ID = re.compile(r"^w\d{3,}$")


class GeneratedWindowError(SystemExit):
    """A generated alchemical directory that cannot be run as it stands."""


def leg_directory(directory: Path) -> Path:
    """`leg/` beside the declaration. One name, so the two surfaces cannot disagree about it."""
    return Path(directory) / "leg"


def require_prepared_leg(directory: Path) -> Path:
    """`leg/leg.json` beside `directory`, or a refusal naming what is missing.

    ONE implementation with TWO callers, and the second one is why it is a function.
    `window_main` runs it, and `md-run` runs it too -- BEFORE it creates `-odir` and writes
    `resolved.config`. Left to `window_main` alone, the refusal would arrive after the output
    directory existed, and a `-odir` holding a `resolved.config` is indistinguishable from a run
    that happened. That is the ordinary case of the mistake this refuses: a mistyped `-odir`.
    """
    leg = leg_directory(directory)
    if not (leg / "leg.json").is_file():
        raise GeneratedWindowError(
            f"{leg}/leg.json does not exist, so this directory declares a ladder whose leg was "
            f"never prepared. `md-openmm build-md` writes it when it generates the run; a run "
            f"directory without it cannot say which end states the windows run between.")
    return leg


def window_ids(resolved: dict[str, Any]) -> list[str]:
    """Every window id of this ladder, in lambda order, from the resolved configuration alone.

    Derived rather than listed: `build-md` names the scripts from this function and the runtime
    reads them from it, so a run directory cannot hold scripts for one set of windows and a
    declaration describing another.
    """
    from ..build.md import alchemical_lambda_values

    return [f"w{index:03d}" for index in range(len(alchemical_lambda_values(resolved)))]


def window_parser(description: str):
    """The flags a window accepts. IN THE RUNTIME, not in the generated file.

    A generated script is an entry point: it names what it runs and imports one function. The
    flags belong here for the same reason `stage_parser` lives in `md_tools.md.stage` -- so
    `python w003.py --cpu` and `md-openmm md-run -i alchemical.in --window w003 --cpu` are the
    same parser and cannot come to mean different things.
    """
    import argparse

    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--cpu", action="store_true",
                        help="run on the CPU platform. The one per-run override; the machine "
                             "configuration decides otherwise.")
    parser.add_argument("--device", type=int, default=None, help="CUDA device index")
    parser.add_argument("--machine-config", default=None,
                        help="machine configuration; $MD_TOOLS_CONFIG or the user file otherwise")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace this window's outputs; the previous ones are moved aside")
    parser.add_argument("--check", action="store_true",
                        help="validate and create nothing")
    return parser


def select_windows(resolved: dict[str, Any], selector: str | None) -> list[str]:
    """The windows a selector names: `w003`, `3`, `0-5`, `0,3,7`, or every window when None.

    Mirrors AIS's `--paths` deliberately: a person who has run switching paths should not have to
    learn a second notation to run a subset of windows.
    """
    every = window_ids(resolved)
    if selector is None:
        return every
    chosen: list[str] = []
    for part in str(selector).replace(" ", "").split(","):
        if not part:
            continue
        if WINDOW_ID.match(part):
            names = [part]
        elif "-" in part.lstrip("-"):
            low, _, high = part.partition("-")
            try:
                span = range(int(low), int(high) + 1)
            except ValueError:
                raise GeneratedWindowError(
                    f"--window {selector!r}: {part!r} is not a window, an index or a range of "
                    f"indices. Windows are {every[0]}..{every[-1]}, or 0-{len(every) - 1}.")
            names = [f"w{index:03d}" for index in span]
        else:
            try:
                names = [f"w{int(part):03d}"]
            except ValueError:
                raise GeneratedWindowError(
                    f"--window {selector!r}: {part!r} is not a window, an index or a range of "
                    f"indices. Windows are {every[0]}..{every[-1]}, or 0-{len(every) - 1}.")
        for name in names:
            if name not in every:
                raise GeneratedWindowError(
                    f"--window {selector!r} names {name}, and this ladder has "
                    f"{len(every)} window(s): {every[0]}..{every[-1]}. A window that does not "
                    f"exist is refused rather than skipped, because a campaign is complete when "
                    f"every window is, and silently running fewer would report a complete "
                    f"campaign that sampled part of the path.")
            if name not in chosen:
                chosen.append(name)
    if not chosen:
        raise GeneratedWindowError(f"--window {selector!r} names no window.")
    return chosen


def window_settings(resolved: dict[str, Any]):
    """`WindowSettings` from the resolved declaration. The runtime owns every rule about it.

    `build-md` has already constructed one of these to validate the configuration; this builds the
    same object from the same keys, so the settings a window runs under are the settings the
    generation refused or accepted, not a second reading of them.
    """
    from .windows import WindowSettings

    block = resolved["alchemical"]
    dynamics = resolved["dynamics"]
    return WindowSettings(
        steps=int(block["window_steps"]),
        report_interval=int(block["report_interval_steps"]),
        checkpoint_interval=int(block["checkpoint_interval_steps"]),
        equilibration_steps=int(block["equilibration_steps"]),
        minimize_iterations=int(block["minimize_iterations"]),
        timestep_fs=dynamics["timestep_fs"],
        friction_per_ps=float(dynamics["friction_per_ps"]),
        barostat_interval=int(dynamics["barostat_interval_steps"]),
        seed=int(dynamics["seed"]))


def _plan_directory(directory: Path, resolved: dict[str, Any]) -> Path:
    """The copied-in plan, by the name `build-md` gave it. Never a path outside the run."""
    stated = (resolved["alchemical"].get("plan") or "").strip()
    local = Path(directory) / stated
    if stated and local.is_dir():
        return local
    copies = sorted(p for p in Path(directory).glob("plan.*") if p.is_dir())
    if len(copies) == 1:
        return copies[0]
    raise GeneratedWindowError(
        f"{directory} holds {len(copies)} copied-in plan directories, and "
        f"alchemical.plan = {stated!r} does not name one of them. `build-md` copies the plan in "
        f"content-addressed so the run directory is movable; regenerate the run rather than "
        f"pointing it at a plan somewhere else.")


def hamiltonian_from(directory: Path, resolved: dict[str, Any]):
    """Rebuild the Hamiltonian from the plan this run was generated from.

    Rebuilt, then checked: `run_leg` compares its System against the `system_sha256` recorded when
    `build-md` wrote `leg/system.xml`, so a rebuild that differs is refused rather than run.
    """
    from .hamiltonian import from_plan
    from .softcore import SoftcoreSettings
    from .topology import load_plan

    block = resolved["alchemical"]
    settings = SoftcoreSettings.from_mapping(
        {key: block[key] for key in ("sc", "softcore_function", "scalpha", "scbeta",
                                     "sc_boundary_14") if key in block})
    plan = load_plan(_plan_directory(directory, resolved))
    return plan, from_plan(plan, settings)


def window_main(resolved: dict[str, Any], *, directory: Path, windows: Sequence[str],
                cpu: bool = False, device: Any = None, machine_config: Any = None,
                overwrite: bool = False, check: bool = False) -> int:
    """Run the named windows of the ladder declared in `resolved`. THE one dispatch.

    `directory` holds `resolved.config`, the copied-in plan and `leg/`. Every window writes into
    `leg/windows/<repeat>/`, and a window that already completed and verifies is skipped by
    `run_window` rather than rerun.
    """
    from .campaign import run_leg

    directory = Path(directory)
    leg = require_prepared_leg(directory)
    _, hamiltonian = hamiltonian_from(directory, resolved)
    settings = window_settings(resolved)
    results = run_leg(leg, hamiltonian=hamiltonian, settings=settings, windows=list(windows),
                      # ONE repeat per run directory, always `r1`. Independent repeats are
                      # independent RUNS -- `alchemical-run2` with its own seed -- because a run
                      # directory is an identity and two repeats sharing one would put two
                      # experiments over one set of paths. `campaign.combine_repeats` combines
                      # them afterwards, from directories that each say what they were.
                      repeat="r1",
                      cpu=cpu, device=device, machine_config=machine_config,
                      overwrite=overwrite, check=check)
    for result in results:
        print(f"{result['window_id']}: {result['disposition']}"
              + (f", {result['rows']} rows" if result.get("rows") is not None else ""))
    return 0


def run_generated_window(script: str | Path, window_id: str, argv: list[str] | None = None) -> int:
    """Run ONE window: the whole body of a generated `w003.py`.

        from md_tools.alchemy import run_generated_window
        raise SystemExit(run_generated_window(__file__, "w003"))

    The window id is a literal in the generated file rather than an argument the script parses,
    for the same reason a stage script carries its own name: a generated entry point declares what
    it runs, and a script that took it from the command line could be pointed at another window
    while still reporting its own.
    """
    from ..build.md import resolve_md_config
    from ..md.stage import resolved_config_beside

    config_path = resolved_config_beside(script)
    resolved = resolve_md_config(config_path)
    if resolved["protocol"] != "alchemical":
        raise GeneratedWindowError(
            f"{config_path} declares protocol {resolved['protocol']!r}, but this is an alchemical "
            f"window script. The directory holds a script and a configuration that describe "
            f"different runs; regenerate it.")
    if window_id not in window_ids(resolved):
        raise GeneratedWindowError(
            f"{Path(script).name} runs {window_id}, and the declaration beside it has "
            f"{len(window_ids(resolved))} window(s). The script and the configuration describe "
            f"different ladders; regenerate the run.")
    args = window_parser(f"{Path(script).name}: one fixed-lambda window").parse_args(argv)
    return window_main(resolved, directory=Path(config_path).parent, windows=[window_id],
                       cpu=args.cpu, device=args.device, machine_config=args.machine_config,
                       overwrite=args.overwrite, check=args.check)
