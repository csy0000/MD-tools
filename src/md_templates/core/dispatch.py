"""Explicit dispatch from a template descriptor to the engine provider that implements it.

A table, not a discovery mechanism. `(method, engine)` maps to a module path and the provider object
inside it; nothing scans entry points, nothing imports at module load, and adding an engine means
adding a readable line rather than satisfying a protocol. The campaign forbids a plugin framework
and this is the smallest thing that is not one.

Three failures are kept apart because they need different answers from whoever hits them:

    the descriptor says dispatch is not active     -> the template is metadata; use its cli_command
    no provider is registered for (method, engine) -> this build cannot run that template at all
    the provider is registered but not importable  -> the engine is not installed here

The third is the common one and the easiest to report badly. `md-templates list` works with no
OpenMM installed, so a user can reasonably reach `prepare` without ever having installed one; the
message names the missing package and the template that wanted it, rather than surfacing an
ImportError from four frames down.

Nothing here imports an engine until a provider is actually requested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["DispatchError", "ProviderUnavailable", "DispatchNotActive", "Provider",
           "PROVIDERS", "provider_for", "registered_providers"]


class DispatchError(RuntimeError):
    """A template could not be dispatched to a provider."""


class DispatchNotActive(DispatchError):
    """The descriptor declares that generic dispatch is not active for this template."""


class ProviderUnavailable(DispatchError):
    """The provider exists but cannot be imported here, usually a missing engine install."""


@dataclass(frozen=True)
class ProviderBinding:
    """Where a provider lives, and what it needs installed."""

    module: str
    attribute: str
    requires: tuple[str, ...]
    description: str


#: The whole dispatch table. One line per (method, engine).
PROVIDERS: dict[tuple[str, str], ProviderBinding] = {
    ("conventional-md", "openmm"): ProviderBinding(
        module="md_templates.engines.openmm.provider",
        attribute="PROVIDER",
        requires=("openmm",),
        description="explicit-water conventional MD on OpenMM",
    ),
    ("rest2", "openmm"): ProviderBinding(
        module="md_templates.engines.openmm.provider",
        attribute="PROVIDER",
        requires=("openmm",),
        description="omega-selective REST2 on OpenMM",
    ),
}


def registered_providers() -> dict[tuple[str, str], ProviderBinding]:
    """The table, for `list`-style commands. Importing nothing."""
    return dict(PROVIDERS)


def provider_for(descriptor, *, require_active: bool = True):
    """Resolve the provider object for one descriptor, importing the engine only now."""
    key = (descriptor.method.id, descriptor.engine.id)
    binding = PROVIDERS.get(key)
    if binding is None:
        known = sorted(f"{method}/{engine}" for method, engine in PROVIDERS)
        raise DispatchError(
            f"no provider is registered for method {key[0]!r} on engine {key[1]!r}, so this build "
            f"cannot run {descriptor.template_id!r}. Registered: {known}"
        )

    if require_active and descriptor.implementation.dispatch != "catalog":
        raise DispatchNotActive(
            f"{descriptor.template_id!r} declares dispatch "
            f"{descriptor.implementation.dispatch!r}, so it is catalog metadata rather than an "
            f"activated template. Run it directly with "
            f"`{descriptor.implementation.cli_command}`."
        )

    missing = []
    for requirement in binding.requires:
        try:
            __import__(requirement)
        except ImportError:
            missing.append(requirement)
    if missing:
        raise ProviderUnavailable(
            f"{descriptor.template_id!r} needs {', '.join(missing)} installed to run "
            f"({binding.description}). Listing and inspecting templates works without it; "
            f"executing one does not."
        )

    try:
        module = __import__(binding.module, fromlist=[binding.attribute])
    except ImportError as exc:
        raise ProviderUnavailable(
            f"the provider for {descriptor.template_id!r} ({binding.module}) could not be "
            f"imported: {exc}"
        ) from exc

    provider = getattr(module, binding.attribute, None)
    if provider is None:
        raise DispatchError(
            f"{binding.module} does not define {binding.attribute}, so "
            f"{descriptor.template_id!r} has no usable provider"
        )
    return provider


@dataclass(frozen=True)
class Provider:
    """What a provider must offer. Deliberately tiny.

    `run(action, descriptor, args)` returns a process exit code. Keeping the surface to one call
    means a provider is free to implement `prepare`, `run` and `resume` however its engine works,
    and the generic CLI never grows engine-shaped arguments.
    """

    name: str
    engine: str
    actions: tuple[str, ...]
    _run: Optional[object] = None

    def run(self, action: str, descriptor, args) -> int:  # pragma: no cover - overridden
        raise NotImplementedError
