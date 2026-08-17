"""Freeze the public surface the PR3-PR8 migration must carry across unchanged.

    python scripts/capture_public_surface.py            # write tests/baseline/public_surface.json
    python scripts/capture_public_surface.py --check    # compare without writing

The compatibility goldens under `tests/goldens/` freeze *scientific* contracts -- hashes, profiles,
seeds, ladders. This freezes the *interface* contracts, which a large structural migration is far
more likely to break by accident: a command that loses an option, an exception that stops being
importable from where it always was, a default that changes because a parser moved.

Recorded deliberately as structure rather than prose:

  cli        every command and subcommand, and for each one every option string, destination,
             default, choice list and required flag. Comparing rendered help text alone would tie
             the baseline to wording; comparing the parse tree ties it to behaviour.
  imports    every public name reachable from the legacy namespaces, and the module each one comes
             from -- so a symbol that silently moves is visible even when `from x import y` keeps
             working through a re-export.
  exceptions the full MRO of every public error type, because `except` clauses in consuming code
             depend on the inheritance chain, not just the name.

Deterministic: no timestamps, no paths, no environment. Two runs on one commit produce identical
bytes.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "tests" / "baseline"
BASELINE_FILE = BASELINE_DIR / "public_surface.json"


def _action_record(action) -> dict:
    default = action.default
    if isinstance(default, (str, int, float, bool, type(None))):
        recorded = default
    else:
        recorded = f"<{type(default).__name__}>"
    return {
        "option_strings": list(action.option_strings),
        "dest": action.dest,
        "required": bool(action.required),
        "default": recorded,
        "choices": sorted(map(str, action.choices)) if action.choices else None,
        "nargs": action.nargs if isinstance(action.nargs, (str, int, type(None))) else str(action.nargs),
        "type": getattr(action.type, "__name__", None) if action.type else None,
    }


def _walk_parser(parser, path: str, out: dict) -> None:
    actions = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in sorted(action.choices.items()):
                _walk_parser(subparser, f"{path} {name}".strip(), out)
            continue
        if action.dest == "help":
            continue
        actions.append(_action_record(action))
    out[path] = {
        "prog": parser.prog,
        "actions": sorted(actions, key=lambda a: (a["dest"], a["option_strings"])),
    }


def _cli_surface() -> dict:
    from md_templates.openmm.cli import build_parser

    out: dict = {}
    _walk_parser(build_parser(), "md-openmm", out)
    return out


def _module_surface(module_name: str) -> dict:
    module = __import__(module_name, fromlist=["*"])
    names = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        value = getattr(module, name)
        origin = getattr(value, "__module__", None)
        if inspect.ismodule(value):
            kind, origin = "module", value.__name__
        elif inspect.isclass(value):
            kind = "class"
        elif callable(value):
            kind = "callable"
        else:
            kind = type(value).__name__
        names[name] = {"kind": kind, "origin": origin}
    return names


def _exception_surface() -> dict:
    import md_templates.core as core

    modules = ["md_templates.openmm.runstate", "md_templates.openmm.bundlev2",
               "md_templates.openmm.schemas", "md_templates.openmm.spec.resolve",
               "md_templates.openmm.spec.units", "md_templates.openmm.spec.migrate"]
    out: dict = {}
    for module_name in modules:
        module = __import__(module_name, fromlist=["*"])
        for name in dir(module):
            value = getattr(module, name)
            if inspect.isclass(value) and issubclass(value, BaseException) \
                    and value.__module__ == module_name:
                out[f"{module_name}.{name}"] = [b.__name__ for b in value.__mro__]
    for name in core.__all__:
        value = getattr(core, name)
        if inspect.isclass(value) and issubclass(value, BaseException):
            out[f"md_templates.core.{name}"] = [b.__name__ for b in value.__mro__]
    return out


def _packaged_modules() -> list[str]:
    """Every importable module path under the package, so a move cannot go unnoticed."""
    base = REPO_ROOT / "src" / "md_templates"
    out = []
    for path in sorted(base.rglob("*.py")):
        relative = path.relative_to(base.parent)
        if "__pycache__" in relative.parts or "_packaged" in relative.parts:
            continue
        dotted = ".".join(relative.with_suffix("").parts)
        out.append(dotted.removesuffix(".__init__"))
    return sorted(set(out))


def build_surface() -> dict:
    return {
        "cli": _cli_surface(),
        "imports": {
            "md_templates.openmm": _module_surface("md_templates.openmm"),
            "md_templates.core": _module_surface("md_templates.core"),
        },
        "exceptions": _exception_surface(),
        "modules": _packaged_modules(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compare against the committed baseline instead of writing it")
    args = parser.parse_args(argv)

    payload = build_surface()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.check:
        BASELINE_FILE.write_text(text, encoding="utf-8")
        print(f"written {BASELINE_FILE.relative_to(REPO_ROOT)}: "
              f"{len(payload['cli'])} command(s), "
              f"{sum(len(v) for v in payload['imports'].values())} public name(s), "
              f"{len(payload['exceptions'])} exception(s), {len(payload['modules'])} module(s)")
        return 0

    if not BASELINE_FILE.is_file():
        print(f"missing {BASELINE_FILE}", file=sys.stderr)
        return 1
    if BASELINE_FILE.read_text(encoding="utf-8") != text:
        print("public surface DIFFERS from the committed baseline", file=sys.stderr)
        return 1
    print("public surface: unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
