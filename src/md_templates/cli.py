"""`md-templates`: the generic, engine-neutral command over the template catalog.

Three of its commands — `list`, `inspect`, `identity` — must work anywhere: a laptop, a CI lint job,
a machine that could never run a simulation. They therefore import nothing heavier than YAML and
pydantic, and a test proves it by running them in a subprocess where OpenMM is unimportable.

The execution commands — `prepare`, `run`, `resume` — do need an engine, and they say so when it is
missing rather than failing with an import traceback. Dispatch is **explicit**: the template
descriptor names its provider, and `md_templates.core.dispatch` maps `(method, engine)` to it. There
is no plugin discovery, no entry-point scanning and no import-time side effects; adding an engine
means adding a line to a table someone can read.

`md-openmm` is unchanged and remains the supported way to drive the OpenMM implementation directly.
This command is the layer above it: it knows which templates exist and which provider serves each
one, and it hands the work over.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3


def _catalog(args):
    """The catalog to work with, chosen without ever consulting the working directory.

    `--repository` is the explicit override. Otherwise the root comes from
    `md_templates.core.config.sources.catalog_root()`: the catalog packaged into this installation,
    or the repository this package is being imported from.

    Deliberately the *same* function that profile discovery uses. An earlier version walked up from
    `cwd` here while profiles walked up from the package, and the two disagreed the moment a command
    ran from an unrelated directory -- the catalog was "not found" while its profiles resolved
    perfectly. One source of truth or the two drift apart under exactly the conditions the packaging
    work exists to support.
    """
    from md_templates.core import load_catalog
    from md_templates.core.config.sources import catalog_root
    from md_templates.core.packaged import PackagedCatalogError, load_packaged_catalog

    if args.repository:
        root = Path(args.repository)
        return load_catalog(root), root

    root = catalog_root()
    if root is None:
        raise SystemExit(
            "no catalog found: this installation carries no packaged catalog and is not being "
            "imported from a repository. Pass --repository DIR."
        )
    if root.name == "_packaged":
        try:
            return load_packaged_catalog(), None
        except PackagedCatalogError as exc:
            raise SystemExit(f"the packaged catalog is unusable: {exc}")
    return load_catalog(root), root


def cmd_list(args) -> int:
    catalog, root = _catalog(args)
    if args.format == "json":
        print(json.dumps([{
            "template_id": e.template_id,
            "method": e.method,
            "engine": e.engine,
            "template_path": e.template_path,
            "aliases": list(e.method_aliases),
            "tags": list(e.tags),
            "dispatch": catalog.descriptor(e.template_id).implementation.dispatch,
            "scientific_status": catalog.descriptor(e.template_id).validation.scientific_status,
        } for e in catalog.entries], indent=2))
        return EXIT_OK

    print(f"catalog: {root or 'packaged with this installation'}")
    for entry in catalog.entries:
        descriptor = catalog.descriptor(entry.template_id)
        print(f"  {entry.template_id}")
        print(f"      method {entry.method}"
              + (f" (aliases: {', '.join(entry.method_aliases)})" if entry.method_aliases else "")
              + f"   engine {entry.engine}")
        print(f"      dispatch {descriptor.implementation.dispatch}"
              f"   implementation {descriptor.validation.implementation_status}"
              f"   science {descriptor.validation.scientific_status}")
    return EXIT_OK


def cmd_inspect(args) -> int:
    catalog, _ = _catalog(args)
    descriptor = catalog.descriptor(args.template)
    if args.format == "json":
        print(json.dumps(descriptor.model_dump(mode="json"), indent=2))
        return EXIT_OK

    print(f"{descriptor.template_id}  ({descriptor.display_name})")
    print(f"  {descriptor.summary.strip()}")
    print(f"  method    {descriptor.method.id} v{descriptor.method.api_version}"
          f"   engine {descriptor.engine.id} v{descriptor.engine.api_version}")
    print(f"  provider  {descriptor.implementation.python_namespace}"
          f"   dispatch {descriptor.implementation.dispatch}"
          f"   cli {descriptor.implementation.cli_command}")
    print(f"  routes    {', '.join(r.id for r in descriptor.input_routes)}")
    print(f"  bundles   reads {descriptor.bundle_schema.readable}"
          f"  writes {descriptor.bundle_schema.writable}")
    print(f"  profiles  {descriptor.profiles.provider}"
          f"  template-local={descriptor.profiles.template_local}")
    for profile_id in descriptor.profiles.profile_ids:
        print(f"              {profile_id}")
    if descriptor.method_features:
        print("  features")
        for feature in descriptor.method_features:
            print(f"              {feature.id}: supported={feature.supported} "
                  f"default_enabled={feature.default_enabled} disableable={feature.disableable}")
    print(f"  status    implementation {descriptor.validation.implementation_status}"
          f"   scientific {descriptor.validation.scientific_status}")
    for gate in descriptor.validation.open_gates:
        print(f"      open gate: {gate}")
    return EXIT_OK


def cmd_identity(args) -> int:
    from md_templates.core.identity import IdentityError, resolve_identity
    from md_templates.core.packaged import resolve_packaged_identity

    catalog, root = _catalog(args)
    try:
        identity = (resolve_identity(catalog, args.template, root=root) if root is not None
                    else resolve_packaged_identity(args.template))
    except IdentityError as exc:
        print(f"no immutable identity: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    if args.format == "json":
        print(json.dumps(identity.as_dict(), indent=2))
    else:
        print(identity.canonical)
    return EXIT_OK


def _dispatch(args, action: str) -> int:
    from md_templates.core.dispatch import DispatchError, provider_for

    catalog, _ = _catalog(args)
    entry = catalog.require(args.template)
    descriptor = catalog.descriptor(entry.template_id)
    try:
        provider = provider_for(descriptor)
    except DispatchError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_UNAVAILABLE
    return provider.run(action, descriptor, args)


def cmd_prepare(args) -> int:
    return _dispatch(args, "prepare")


def cmd_run(args) -> int:
    return _dispatch(args, "run")


def cmd_resume(args) -> int:
    return _dispatch(args, "resume")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="md-templates",
        description="List, inspect and identify simulation templates, and dispatch runs to the "
                    "engine provider that implements them.",
    )
    p.add_argument("--repository", metavar="DIR",
                   help="the repository root holding registry.yaml. Defaults to the catalog "
                        "packaged with this installation, or the repository this package is "
                        "imported from. Never derived from the working directory.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_format(parser, choices=("text", "json")):
        parser.add_argument("--format", choices=list(choices), default="text")

    ls = sub.add_parser("list", help="every template this catalog publishes")
    add_format(ls)
    ls.set_defaults(func=cmd_list)

    ins = sub.add_parser("inspect", help="one template's declared contract and status")
    ins.add_argument("template", help="template id or logical template path")
    add_format(ins)
    ins.set_defaults(func=cmd_inspect)

    idy = sub.add_parser("identity", help="the exact immutable identity of one template")
    idy.add_argument("template", help="template id or logical template path")
    add_format(idy)
    idy.set_defaults(func=cmd_identity)

    # execution: the provider owns the remaining arguments, so they are passed through unchanged
    for name, func, helptext in (
        ("prepare", cmd_prepare, "build a bundle for this template"),
        ("run", cmd_run, "run production for this template from a prepared bundle"),
        ("resume", cmd_resume, "continue an interrupted run"),
    ):
        ex = sub.add_parser(name, help=helptext)
        ex.add_argument("--template", required=True, help="template id or logical template path")
        ex.add_argument("provider_args", nargs=argparse.REMAINDER,
                        help="arguments passed through to the engine provider unchanged")
        ex.set_defaults(func=func)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
