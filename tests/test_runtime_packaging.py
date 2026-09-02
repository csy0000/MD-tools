"""Every module a generated script reaches is a real package module, and ships as one.

This replaces `test_generated_project_completeness.py`, which computed the executor's import
closure over a directory of loose modules that were put on `sys.path` by hand. That test existed
because a module missing from that directory failed only at run time and only on the path that
imported it -- an ordinary package cannot be half-shipped that way, and setuptools decides what is
included rather than a hand-maintained list.

What replaced it is the property that made the closure unnecessary: there are no bare imports left
to resolve, and no directory to put on `sys.path`.

PLATFORM_POLICY_EXEMPTION: static analysis only. Nothing runs.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "md_tools"
RUNTIME_PACKAGES = ("md", "rest2", "remd", "ais")

#: The loose-module names the runtime used to import by bare name. None may come back: a bare
#: import resolves only when something has put a directory on `sys.path`, which is the mechanism
#: this migration removed -- and while it was in place `remd/statistics.py` shadowed the standard
#: library's `statistics` for anything that imported it.
RETIRED_BARE_MODULES = (
    "replica_driver", "replica_engine", "replica_protocol", "replica_runtime", "replica_schedule",
    "replica_statistics", "replica_storage", "replica_validate", "replica_executor",
    "exchange_rules", "rrest2_reservoir", "source_ensemble", "rem_log", "amber_trajectory",
    "state_trajectories", "md_stages", "phase_space", "rest2_scaling", "hamiltonian_identity",
)


def _runtime_modules():
    for package in RUNTIME_PACKAGES:
        yield from sorted((SRC / package).glob("*.py"))


def test_every_runtime_package_is_a_real_package():
    for package in RUNTIME_PACKAGES:
        directory = SRC / package
        assert directory.is_dir(), f"md_tools.{package} is missing"
        assert (directory / "__init__.py").is_file(), f"md_tools.{package} has no __init__.py"


def test_no_runtime_module_imports_another_by_bare_name():
    """A bare import needs a directory on `sys.path`; a package import does not.

    The one deliberate exception is a file loaded BY PATH -- the `--exchange-rule` plug-in -- which
    is not part of any package when it runs and must therefore import absolutely.
    """
    offenders = []
    for path in _runtime_modules():
        if path.name == "rrest2_exchange.py":
            continue                                   # loaded by path; see the docstring above
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.split(".")[0] in RETIRED_BARE_MODULES:
                    offenders.append(f"{path.name}: from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in RETIRED_BARE_MODULES:
                        offenders.append(f"{path.name}: import {alias.name}")
    assert not offenders, "bare imports are back:\n  " + "\n  ".join(offenders)


def test_the_retired_templates_directory_is_gone():
    """It held installed runtime code under a name that said it held templates."""
    assert not (SRC / "openmm" / "templates").exists()


def test_no_runtime_module_shadows_a_standard_library_module():
    """`remd/statistics.py` shadowed the stdlib `statistics` while the directory was on sys.path.

    It does not any more, because nothing puts the directory there -- but the name is still
    `statistics.py`, so this asserts the property that makes that safe rather than trusting it.
    """
    import sys

    for path in _runtime_modules():
        assert str(path.parent) not in sys.path, (
            f"{path.parent} is on sys.path; its modules shadow anything with the same name")


@pytest.mark.parametrize("package, expected", [
    ("md", {"run_generated_stage", "run_generated_workflow", "run_stage",
            "PositionalRestraint", "ReportingConfig"}),
    ("rest2", {"REST2Scaler", "ScalingSelection"}),
    ("remd", {"REMDRunner", "NeighborExchangeRule", "run_remd", "run_generated_remd"}),
    ("ais", {"run_generated_ais"}),
])
def test_each_package_exports_the_documented_names(package, expected):
    import importlib

    module = importlib.import_module(f"md_tools.{package}")
    assert expected <= set(module.__all__), expected - set(module.__all__)
    for name in expected:
        assert hasattr(module, name), name


def test_the_reservoir_rule_is_exposed_generically():
    """rREST2 composes it; a future reservoir REMD method must be able to reuse it."""
    from md_tools.remd.reservoir import ReservoirRefreshRule

    assert hasattr(ReservoirRefreshRule, "propose")
    assert hasattr(ReservoirRefreshRule, "describe")


# --- the compatibility facades ------------------------------------------------------------------

def test_the_runtime_facades_reexport_and_carry_no_logic():
    """A pre-v0.5 generated script still runs, and there is no second authority to drift.

    The facades are permitted *because* they contain no scientific logic: each name is the same
    object the new API exposes, so a correction lands once and both entry points see it. This
    asserts identity, not merely that an import succeeds.
    """
    from md_tools.ais import run_generated_ais
    from md_tools.md import run_generated_stage, run_stage
    from md_tools.remd import run_generated_remd
    from md_tools.runtime import ais as facade_ais
    from md_tools.runtime import replica as facade_replica
    from md_tools.runtime import stage as facade_stage

    assert facade_stage.run_stage is run_stage
    assert facade_stage.run_generated_stage is run_generated_stage
    assert facade_replica.run_generated_remd is run_generated_remd
    assert facade_ais.run_generated_ais is run_generated_ais


def test_no_facade_defines_anything_of_its_own():
    """A facade with a function in it is a fork waiting to happen."""
    import ast

    for name in ("stage.py", "replica.py", "ais.py", "__init__.py"):
        path = SRC / "runtime" / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = [n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        assert not defined, f"runtime/{name} defines {defined}; it must only re-export"


def test_nothing_in_the_package_imports_the_compatibility_facades():
    """They exist for already-generated scripts, not for this package to use."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.parent.name == "runtime":
            continue
        text = path.read_text(encoding="utf-8")
        for form in ("from ..runtime", "from .runtime", "md_tools.runtime"):
            if form in text:
                offenders.append(f"{path.relative_to(SRC)}: {form}")
    assert not offenders, offenders


# --- one implementation of each low-level operation ---------------------------------------------

def test_there_is_one_file_digest_implementation():
    """It lived in four modules. All four produced the same digest, which is exactly why nobody
    noticed there were four."""
    definitions = sorted(p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
                         if "def sha256_file" in p.read_text(encoding="utf-8"))
    assert definitions == ["build/record.py"], definitions


def test_the_two_seed_derivations_have_different_names():
    """They are different algorithms for different jobs, and shared one name until v0.5.

    The run-time one cannot be changed to match the build-time one: AIS selects its starting
    frames from it, so a different derivation would silently change which configurations every
    existing AIS project starts from. So they keep two implementations -- and two names.
    """
    from md_tools.md import derive_seed
    from md_tools.openmm.seeds import derive_build_seed

    assert derive_seed(7, "ions") != derive_build_seed(7, "ions"), (
        "the two derivations now agree, so one of them changed; check which projects that moves")
    definitions = sorted(p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
                         if "def derive_seed" in p.read_text(encoding="utf-8"))
    assert definitions == ["md/_stages.py"], definitions


def test_there_is_no_second_platform_resolver_left():
    """MIGRATED. This asserted that the TWO platform functions had different names.

    There is one now. `remd.engine.build_platform` is gone: it resolved `None`/"automatic" with
    `"CUDA" if "CUDA" in available else "CPU"`, an automatic CPU fallback in the one place a
    stage's refusal could not reach, and a ladder that fell onto the CPU that way still ran,
    still wrote a trajectory and still reported success two orders of magnitude later. The driver
    consumes the platform `preflight_ladder` resolved, so there was nothing left to fall back
    from and nothing left calling it.

    What is asserted instead is the stronger property: no module defines a second resolver, and
    no module still contains that fallback expression.
    """
    from md_tools.md import resolve_platform

    assert callable(resolve_platform)
    import md_tools.remd.engine as engine

    assert not hasattr(engine, "build_platform"), "the second platform resolver is back"
    # CODE only. Comments and docstrings explaining the removal must be allowed to quote the
    # expression they are explaining -- a guard that forbids naming the defect forbids the
    # explanation with it, and the explanation is what stops someone reintroducing it.
    import io
    import tokenize

    for module in SRC.rglob("*.py"):
        code = []
        with tokenize.open(module) as handle:
            for token in tokenize.generate_tokens(handle.readline):
                if token.type not in (tokenize.COMMENT, tokenize.STRING):
                    code.append(token.string)
        assert 'if"CUDA"inavailableelse"CPU"' not in "".join(code).replace(" ", ""), (
            f"{module.relative_to(SRC)} falls back to the CPU automatically")
    definitions = sorted(p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
                         if "def resolve_platform(" in p.read_text(encoding="utf-8"))
    # `openmm/platform_policy.resolve_platform_request` is a different name for a different job:
    # it is the CENTRAL policy, and `md/_stages.resolve_platform` is the small name-only helper
    # it superseded for stages. Matched on the exact `def resolve_platform(` so the substring
    # does not collide.
    assert definitions == ["md/_stages.py"], definitions
