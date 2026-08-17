"""Where versioned default profiles come from, once a template directory can own them.

Phase 5 makes each template directory authoritative for its own profiles. That raises one question
with a lot of wrong answers: how does the resolver *find* them, from a source checkout and from an
installed wheel, without either duplicating the files or depending on the working directory?

**Not from the working directory.** A resolver that walks up from `cwd` looking for `registry.yaml`
would resolve different profiles depending on where the user happened to stand — and would silently
find nothing when a wheel-installed run executes from an unrelated directory, which is exactly the
case the packaging work exists to support.

The roots are derived from the *package's own location* instead, which is fixed at install time:

1. the catalog packaged into this distribution, if there is one (`core/_packaged/`);
2. otherwise the repository this package is being imported from, found by walking up from
   `md_templates/__init__.py` until a `registry.yaml` appears — true for a checkout on `PYTHONPATH`
   or an editable install, and false for a wheel, which is why (1) comes first.

Both give the same files: the build copies the template directories into the packaged catalog from
the tracked source, so there is one authoritative copy and one derived one, never two sources.

Built-in profiles under `core/config/profiles/` remain first in the search order and are still the
only place a profile may live if no template claims it. Duplicate profile IDs across sources are an
error rather than a precedence puzzle: two files claiming one ID means one of them is being ignored,
and which one would depend on directory ordering.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["profile_directories", "template_profile_directories", "catalog_root"]

#: `templates/<method>/<engine>/<variant>/profiles`
_TEMPLATE_PROFILE_GLOB = "templates/*/*/*/profiles"


def _packaged_root() -> Path | None:
    try:
        from ..packaged import packaged_catalog_root

        return packaged_catalog_root()
    except Exception:                                                # noqa: BLE001
        return None


def _source_root() -> Path | None:
    """The repository this package is imported from, if it is one. Never uses `cwd`."""
    import md_templates

    here = Path(md_templates.__file__).resolve().parent
    for candidate in here.parents:
        if (candidate / "registry.yaml").is_file():
            return candidate
    return None


def catalog_root() -> Path | None:
    """The root whose `templates/` tree this installation should read profiles from."""
    return _packaged_root() or _source_root()


def template_profile_directories() -> list[Path]:
    """Every template-local `profiles/` directory this installation can see."""
    root = catalog_root()
    if root is None:
        return []
    return sorted(p for p in root.glob(_TEMPLATE_PROFILE_GLOB) if p.is_dir())


def profile_directories(builtin: Path) -> list[Path]:
    """The full search order: the built-in directory, then every template-local one."""
    return [builtin, *template_profile_directories()]
