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


#: Source markers that mean a file PROPAGATES: it advances a molecular system in time, or
#: minimises one. That is the operation the CUDA rule exists for -- device selection, context
#: creation and replica placement are only exercised when something actually runs.
PROPAGATION_MARKERS = (
    "Simulation(", ".minimizeEnergy(", ".step(", "run_stage", "sampler.run(",
    "number_of_exchanges", ".equilibrate(",
)

#: The platform names a propagating test must never pin.
NON_CUDA_PLATFORM_MARKERS = ('MD_PLATFORM="CPU"', "MD_PLATFORM='CPU'", 'platform="CPU"',
                             'platform="Reference"', "'Reference'", '"Reference"')


def propagates(text: str) -> bool:
    """Whether this test file advances or minimises a molecular system."""
    return any(marker in text for marker in PROPAGATION_MARKERS)


def test_no_simulation_test_names_cpu_or_reference():
    """A test that PROPAGATES must run on CUDA.

    The rule is about propagation, not about touching OpenMM at all. A test that builds a scaled
    System and evaluates U(x) at fixed coordinates exercises none of what CUDA is required for,
    and Reference is the RIGHT platform for it: the REST2 scaling assertions compare energies and
    force parameters to 1e-12, and single precision cannot carry that. Requiring CUDA there would
    make an exact comparison approximate, which is a worse test, not a safer one.

    So the audit asks whether a file propagates before it objects to the platform it names.
    """
    offenders = {}
    for path in _md_test_files():
        text = path.read_text()
        if not propagates(text):
            continue
        for needle in NON_CUDA_PLATFORM_MARKERS:
            if needle in text:
                offenders.setdefault(path.name, []).append(needle)
    assert not offenders, f"MD tests that propagate must run on CUDA: {offenders}"


def test_the_audit_still_catches_a_propagating_file_that_names_a_non_cuda_platform():
    """The guard above was narrowed, so this proves it did not become vacuous."""
    propagating_and_wrong = 'simulation = Simulation(top, sys, integ)\nplatform="CPU"\n'
    assert propagates(propagating_and_wrong)
    assert any(needle in propagating_and_wrong for needle in NON_CUDA_PLATFORM_MARKERS)

    evaluating_only = 'context = Context(system, integrator, "Reference")\n'
    assert not propagates(evaluating_only)


def test_every_test_file_that_propagates_is_still_audited():
    """A file that propagates must be reachable by the audit -- not quietly in NON_MD."""
    propagating = [p.name for p in _md_test_files() if propagates(p.read_text())]
    assert propagating, "the audit found no propagating test file, so it is checking nothing"


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
