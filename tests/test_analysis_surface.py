"""What `md_tools.analysis` promises about itself: its cost, its independence, and its MI estimator.

The contract tests for the clustering estimator are in `test_analysis_t_hdbscan_contract.py` and
came from the hpREST2 session that wrote it. This file tests the PORT rather than the estimator:
the claims that are true of this package and were not true of theirs, plus the mutual-information
module, which arrived without tests of its own.

Each claim here is one the install page or a module docstring makes in prose. A docstring is not a
guard; these are.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "md_tools" / "analysis"


def _in_subprocess(code: str) -> str:
    """Run `code` in a FRESH interpreter. Import cost cannot be measured in this one: pytest has
    already imported half the tree, and `sys.modules` would answer about the session."""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(ROOT), env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin"})
    assert out.returncode == 0, f"subprocess failed:\n{out.stderr}"
    return out.stdout.strip()


def test_importing_the_package_does_not_import_sklearn():
    """The whole reason scikit-learn is an EXTRA. A simulation user installs no ML stack, and an
    eager import here would make `import md_tools` fail for them -- or, worse, succeed slowly."""
    assert _in_subprocess(
        "import sys, md_tools.analysis; print('sklearn' in sys.modules)") == "False"


def test_naming_the_public_estimators_still_does_not_import_sklearn():
    """`from md_tools.analysis import t_hdbscan` resolves the CLASS, which must not drag the
    fitting backend in with it. Without this the lazy `__getattr__` could be defeated by one
    module-level import added later in t_hdbscan.py, and nothing would notice."""
    assert _in_subprocess(
        "import sys\n"
        "from md_tools.analysis import t_hdbscan, torsional_mi\n"
        "print('sklearn' in sys.modules)") == "False"


def test_the_mi_module_imports_without_sklearn_at_all():
    """`torsional_mi` is numpy-only and must stay that way: it is documented as needing no extra,
    so an environment with the extra uninstalled must still be able to use it."""
    assert _in_subprocess(
        "import sys, md_tools.analysis._t_mi\n"
        "print('sklearn' in sys.modules, 'scipy' in sys.modules)") == "False False"


def test_neither_analysis_module_imports_the_other():
    """The clustering and the mutual information are INDEPENDENT, which is why the port extracted
    seven functions instead of vendoring a 1600-line module. Asserted on the AST rather than by
    importing, so it holds even for an import added inside a function body.
    """
    for name, forbidden in (("_t_hdbscan.py", "_t_mi"), ("_t_mi.py", "_t_hdbscan")):
        tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and forbidden in (node.module or ""):
                pytest.fail(f"{name} imports {forbidden}; the two modules must stay independent")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert forbidden not in alias.name, f"{name} imports {forbidden}"


def test_the_extraction_left_the_mutual_information_framework_behind():
    """The metric layer must not reach an MI implementation. The extraction computed the
    transitive closure of seven functions and found the MI layer outside it; this says the result
    did not drift back. `_t_mi.py` inlines the two helpers it needs and is exempt."""
    source = (PACKAGE / "_torsions.py").read_text(encoding="utf-8")
    for name in ("redundancy_matrix", "sensitivity_scan", "mi_pair", "_shift_surrogate_bias"):
        assert name not in source, (
            f"_torsions.py mentions {name}: the redundancy/MI layer is creeping back into the "
            f"metric layer, which is what extracting rather than vendoring was for")


# ------------------------------------------------------------------ the mutual information

def _coupled_and_independent(n=6000, seed=0):
    """Three torsions: `a`, `b` determined by `a` plus noise, and `c` independent of both."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-np.pi, np.pi, n)
    b = (a + rng.normal(0.0, 0.25, n) + np.pi) % (2 * np.pi) - np.pi
    c = rng.uniform(-np.pi, np.pi, n)
    return np.column_stack([a, b, c])


def test_the_plugin_estimator_reports_its_own_bias_rather_than_hiding_it():
    """THE REASON THE NULL EXISTS. The plug-in histogram estimator is biased UPWARD, so an
    INDEPENDENT pair still returns a positive MI -- a number that reads as a weak coupling and is
    entirely bias. The debiased value must come back near zero for the independent pair and must
    stay large for the coupled one."""
    from md_tools.analysis import torsional_mi

    out = torsional_mi(_coupled_and_independent(), names=["a", "b", "c"], bins=24, n_null=8)
    pairs = {(p["a"], p["b"]): p for p in out["pairs"]}
    coupled, independent = pairs[("a", "b")], pairs[("a", "c")]

    assert independent["mi_nats"] > 0.0, "a plug-in estimate of an independent pair is not zero"
    assert abs(independent["mi_nats_debiased"]) < independent["mi_nats"], (
        "debiasing must move the independent pair toward zero")
    assert coupled["mi_nats_debiased"] > 10 * abs(independent["mi_nats_debiased"])


def test_no_null_means_the_debiased_value_is_absent_not_equal_to_the_raw_one():
    """`n_null=0` is allowed, and must not silently return the raw value under the debiased name.
    Equal-but-unmeasured is the failure mode that looks like a result."""
    from md_tools.analysis import torsional_mi

    out = torsional_mi(_coupled_and_independent(n=1500), bins=16, n_null=0)
    for pair in out["pairs"]:
        assert pair["mi_nats_debiased"] is None, (
            "with no null measured, the debiased value must be None rather than the raw value")


def test_the_result_carries_numbers_rather_than_a_verdict():
    """No boolean 'clears the null'. The threshold is the caller's scientific choice, and a bool
    collapses '+615 sd' and '+2.5 sd' into the same answer -- the second being exactly where a
    reader needs the number."""
    from md_tools.analysis import torsional_mi

    out = torsional_mi(_coupled_and_independent(n=2000), bins=16, n_null=6)
    pair = out["pairs"][0]
    assert "excess_over_null_in_sd" in pair and isinstance(pair["excess_over_null_in_sd"], float)
    for banned in ("clears_null", "significant", "is_coupled"):
        assert banned not in pair, f"{banned} bakes the caller's threshold into the estimator"
