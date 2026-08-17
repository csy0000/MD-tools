"""The OpenMM provider: what `md-templates prepare|run|resume` hands work to.

Deliberately thin. Every scientific decision, every default, every hash and every artifact is
produced by exactly the code `md-openmm` already runs — this module chooses which of those entry
points a generic action maps to and passes the user's arguments through unchanged.

That is the whole design, and it is what makes the Phase 5 gate provable: if the generic route
assembled its own argument list, or applied its own defaults, the two routes could agree today and
diverge on the next edit. Instead `md-templates prepare --template conventional-md/openmm/
explicit-water -- --config run.yaml` becomes precisely `md-openmm prepare --config run.yaml`, and a
test asserts the resulting bundles carry identical canonical hashes.

The one thing the provider adds is the method discriminator: a template names its method, so the
generic route knows whether `run` means conventional MD or REST2 without the user restating it.
Passing `--template` and then a contradictory subcommand is refused rather than silently resolved.
"""
from __future__ import annotations

import sys
from typing import Optional

__all__ = ["OpenMMProvider", "PROVIDER"]

#: Which `md-openmm` subcommand implements each generic action, per method.
_ACTION_MAP = {
    "conventional-md": {"prepare": "prepare", "run": "md", "resume": "md"},
    "rest2": {"prepare": "prepare", "run": "rest2", "resume": "rest2"},
}


class OpenMMProvider:
    """Maps a generic action onto the engine's own command line."""

    name = "openmm"
    engine = "openmm"
    actions = ("prepare", "run", "resume")

    def command_for(self, action: str, descriptor) -> str:
        method = descriptor.method.id
        try:
            return _ACTION_MAP[method][action]
        except KeyError:
            raise ValueError(
                f"the OpenMM provider has no {action!r} for method {method!r}; "
                f"it implements {sorted(_ACTION_MAP)} with actions {self.actions}"
            ) from None

    def argv_for(self, action: str, descriptor, provider_args: list[str]) -> list[str]:
        """The exact `md-openmm` argument list this action becomes.

        Separated from `run` so a test can assert the translation without executing a simulation --
        the equivalence claim is about the argument list, and checking it should not cost a
        picosecond of MD.
        """
        passthrough = list(provider_args)
        # argparse.REMAINDER keeps a leading `--` when the user separates provider arguments; it is
        # a separator, not an argument, and forwarding it would make `md-openmm` reject the call.
        if passthrough and passthrough[0] == "--":
            passthrough = passthrough[1:]

        command = self.command_for(action, descriptor)
        if action == "resume":
            if not any(a == "--resume-run" or a.startswith("--resume-run=") for a in passthrough):
                raise ValueError(
                    "resume needs --resume-run RUN_NAME so the run to continue is named "
                    "explicitly. Continuing 'the most recent run' is how the wrong directory gets "
                    "extended."
                )
        return [command, *passthrough]

    def run(self, action: str, descriptor, args) -> int:
        from .cli import main as engine_main

        try:
            argv = self.argv_for(action, descriptor, getattr(args, "provider_args", []) or [])
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        print(f"[md-templates] {descriptor.template_id} -> md-openmm {' '.join(argv)}",
              file=sys.stderr)
        return engine_main(argv)


PROVIDER = OpenMMProvider()
