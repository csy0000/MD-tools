"""No test that integrates a molecular system may run on CPU or Reference.

A CPU run of the MD tests exercises a different code path from the one the work is done on --
device selection, context creation, replica placement -- so it is not evidence that the runtime
works. This file audits the suite for that rule rather than trusting it to be followed by hand.
"""
from __future__ import annotations

import ast
from pathlib import Path

from .conftest import REPO_ROOT

TESTS = Path(__file__).resolve().parent

#: Files allowed to name a non-CUDA platform: installation and platform probes, and unit tests
#: that build no molecular system.
NON_MD = {"test_install.py", "test_platform_policy.py", "test_stages.py"}


def _md_test_files():
    return [p for p in sorted(TESTS.glob("test_*.py")) if p.name not in NON_MD]


def test_no_simulation_test_names_cpu_or_reference():
    offenders = {}
    for path in _md_test_files():
        text = path.read_text()
        for needle in ('MD_PLATFORM="CPU"', "MD_PLATFORM='CPU'", 'platform="CPU"',
                       'platform="Reference"', "'Reference'", '"Reference"'):
            if needle in text:
                offenders.setdefault(path.name, []).append(needle)
    assert not offenders, f"MD tests must run on CUDA: {offenders}"


def test_the_shared_runner_leaves_the_platform_to_the_generated_script():
    """`resolve_platform` picking CUDA is itself part of what the runtime tests check."""
    source = (TESTS / "conftest.py").read_text()
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "run_stage")
    body = ast.get_source_segment(source, function)
    assert 'environment.pop("MD_PLATFORM", None)' in body, body[:400]
    default = function.args.defaults[-2] if len(function.args.defaults) >= 2 else None
    assert isinstance(default, ast.Constant) and default.value is None, \
        "run_stage must not default to a platform"


def test_the_release_workflow_runs_no_dynamics():
    """The GitHub runner has no GPU, so a green tick there must not read as scientific validation."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert 'MD_PLATFORM' not in workflow, "the workflow must not select an MD platform"
    for forbidden in ("run_all.sh", "cMD && python run.py", "equilibrate.py"):
        assert f"({forbidden}" not in workflow and f"cd {forbidden}" not in workflow, forbidden
    assert '-m "not gpu"' in workflow, "the workflow must deselect the simulation tests"
    assert "No simulation validation" in workflow, "it must say what it did not validate"
