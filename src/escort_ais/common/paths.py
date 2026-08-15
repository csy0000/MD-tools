"""Central path resolver for the escort-ais project.

`project_root()` is depth-robust (walks up to pyproject.toml/.git), so it works regardless of
where the importing module lives in the package tree. All other helpers derive from it. The data
root is overridable via the ``ESCORT_AIS_DATA_ROOT`` environment variable.

New code should use the project-generic helpers (`data_root`, `results_root`, `reports_root`,
`prepared_system_dir`, `run_dir`). The legacy `data_dir()`/`results_dir()` return the historical
alanine-specific paths unchanged (production-safe) and are DEPRECATED — do not use in new code.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "ESCORT_AIS_DATA_ROOT"


def project_root() -> Path:
    """Return the repository root for this workspace (depth-robust: walk up to pyproject/.git)."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return p.parents[3]  # fallback: src/escort_ais/common/paths.py -> repo root


def data_root(*, require_exists: bool = False) -> Path:
    """Return the data root: ``$ESCORT_AIS_DATA_ROOT`` if set, else ``<repo>/data``.

    With ``require_exists=True`` raise ``FileNotFoundError`` when the resolved root is absent.
    """
    env = os.environ.get(DATA_ROOT_ENV)
    root = Path(env).expanduser().resolve() if env else project_root() / "data"
    if require_exists and not root.exists():
        raise FileNotFoundError(
            f"data root {root} does not exist (set {DATA_ROOT_ENV} or create it)"
        )
    return root


def results_root() -> Path:
    """Generated figures / intermediate analysis products."""
    return project_root() / "results"


def reports_root() -> Path:
    """Human-readable accepted report bundles."""
    return project_root() / "reports"


def prepared_system_dir(system_slug: str) -> Path:
    """Prepared (built) inputs for a system: ``<data_root>/prepared/<slug>``."""
    return data_root() / "prepared" / system_slug


def run_dir(run_id: str) -> Path:
    """Immutable per-run output directory: ``<data_root>/runs/<run_id>``."""
    return data_root() / "runs" / run_id


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- DEPRECATED alanine-specific helpers (behavior unchanged; kept for existing callers) ---
def data_dir() -> Path:
    """DEPRECATED: historical alanine-specific data dir. Use ``data_root()`` / ``prepared_system_dir``."""
    return data_root() / "alanine_dipeptide"


def results_dir() -> Path:
    """DEPRECATED: historical alanine-specific results dir. Use ``results_root()``."""
    return results_root() / "alanine_dipeptide"
