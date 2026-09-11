"""`md_tools.rest2.hamiltonian` imports only OpenMM, and it is the ONE rung implementation.

Both are contracts a reference bundle depends on. The exporter copies this file byte for byte into
every REST2 bundle so that `verify_rungs.py` can rebuild each rung from rung 0 without md_tools; an
import of anything else would break that silently, only in a bundle, only on a machine without
the package. And `scaler.py` must re-export the same objects rather than keep its own copies, or
the driver and the bundle would build rungs with two implementations that merely agree.
"""
from __future__ import annotations

import ast
from pathlib import Path

import md_tools.rest2.hamiltonian as hamiltonian
import md_tools.rest2.scaler as scaler

SOURCE = Path(hamiltonian.__file__)


def test_the_module_imports_only_openmm():
    offenders = []
    for node in ast.walk(ast.parse(SOURCE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            offenders += [alias.name for alias in node.names if alias.name.split(".")[0] != "openmm"]
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] != "openmm":
                offenders.append("." * node.level + (node.module or ""))
    assert not offenders, f"hamiltonian.py imports {offenders}; a bundle carries it without md_tools"


def test_scaler_re_exports_the_same_objects_rather_than_copies():
    for name in ("build_scaled_system", "scaling_for_tau", "audit_force_classes", "clone_system",
                 "_scale_nonbonded", "_scale_torsions", "_scale_cmap", "_scale_customgb",
                 "torsion_exclusion_report", "REST2_IMPLEMENTATION", "UnclassifiedForceError"):
        assert getattr(scaler, name) is getattr(hamiltonian, name), name
