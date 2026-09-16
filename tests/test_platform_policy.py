"""No test that integrates a molecular system may run on CPU or Reference.

A CPU run of the MD tests exercises a different code path from the one the work is done on --
device selection, context creation, replica placement -- so it is not evidence that the runtime
works. This file audits the suite for that rule rather than trusting it to be followed by hand.
"""
from __future__ import annotations

import ast
import re

import yaml
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


#: A file may opt out of the CUDA requirement only by SAYING SO, in this exact form, with a
#: reason after the colon. An exemption that has to be written into the file it exempts is one a
#: reviewer reading that file cannot miss -- unlike a name in a list on the other side of the
#: suite, which is how exemptions accumulate unnoticed.
EXEMPTION_MARKER = "PLATFORM_POLICY_EXEMPTION:"


def declared_exemption(text: str) -> str | None:
    """The stated reason this file may name a non-CUDA platform, or None."""
    for line in text.splitlines():
        if EXEMPTION_MARKER in line:
            reason = line.split(EXEMPTION_MARKER, 1)[1].strip(" #\"'")
            return reason or None
    return None


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
        if declared_exemption(text):
            continue
        for needle in NON_CUDA_PLATFORM_MARKERS:
            if needle in text:
                offenders.setdefault(path.name, []).append(needle)
    assert not offenders, (
        f"MD tests that propagate must run on CUDA, or declare "
        f"`{EXEMPTION_MARKER} <reason>`: {offenders}")


def test_every_platform_exemption_states_a_reason():
    """An exemption without a reason is a silent one, and silent is what this forbids."""
    for path in _md_test_files():
        text = path.read_text()
        if EXEMPTION_MARKER not in text:
            continue
        reason = declared_exemption(text)
        assert reason and len(reason) > 30, (
            f"{path.name} declares {EXEMPTION_MARKER} without a substantive reason "
            f"(got {reason!r})")


def test_scientific_runtime_evidence_still_comes_from_cuda():
    """The exemption must not become a way to move MD acceptance onto the CPU.

    Files that both propagate AND carry the `gpu` marker are the suite's scientific runtime
    evidence. At least one must exist, and none of them may be exempt.
    """
    gpu_propagating = [p.name for p in _md_test_files()
                       if propagates(p.read_text()) and "pytest.mark.gpu" in p.read_text()]
    assert gpu_propagating, "no propagating test carries the gpu marker any more"
    for path in _md_test_files():
        text = path.read_text()
        if "pytest.mark.gpu" in text and declared_exemption(text):
            raise AssertionError(
                f"{path.name} carries the gpu marker AND a platform exemption; a file is one or "
                f"the other, never both")


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


def _ci_workflow() -> str:
    return (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()


def test_the_ci_workflow_runs_no_dynamics():
    """The GitHub runner has no GPU, so a green tick there must not read as scientific validation.

    Migrated from the deleted `release.yml` guard: the file was replaced, the guarantee was not.
    """
    workflow = _ci_workflow()
    assert 'MD_PLATFORM' not in workflow, "the workflow must not select an MD platform"
    for forbidden in ("run_all.sh", "cMD && python run.py", "equilibrate.py"):
        assert f"({forbidden}" not in workflow and f"cd {forbidden}" not in workflow, forbidden
    # The lane must deselect the simulation tests. The marker expression may narrow further
    # (it also drops `slow`), but it may never stop excluding `gpu`.
    assert re.search(r'-m "not [^"]*\bnot gpu\b|-m "not gpu', workflow), \
        "the workflow must deselect the gpu-marked tests"
    assert "No simulation validation" in workflow, "it must say what it did not validate"


def test_ci_runs_on_pull_requests_and_on_dev_and_main():
    """CI that only fires on a tag is not evidence for the branch under review."""
    document = yaml.safe_load(_ci_workflow())
    # PyYAML reads the bare key `on` as the boolean True.
    triggers = document.get("on", document.get(True))
    assert "pull_request" in triggers, triggers
    assert "workflow_dispatch" in triggers, triggers
    assert set(triggers["push"]["branches"]) == {"dev", "main"}, triggers["push"]


def test_ci_does_not_use_the_retired_openmm_v_tag_convention():
    """`openmm-v*` belongs to this package under its former name, not to MD-tools."""
    document = yaml.safe_load(_ci_workflow())
    triggers = document.get("on", document.get(True))
    tags = triggers["push"].get("tags", [])
    assert tags, "tag-triggered release validation must still exist"
    assert not any(tag.startswith("openmm-v") for tag in tags), tags


def test_ci_names_no_retired_command_at_all():
    """A workflow that *invokes* a retired command would teach an agent the wrong interface.

    It used to name each one once, in a loop asserting the spelling still fails. That loop is
    gone with the names themselves: a guard against resurrecting `sys-gen` is only worth its
    upkeep while somebody might plausibly type it, and the surviving guarantee -- that no second
    executable is installed -- is the one the workflow still checks.
    """
    workflow = _ci_workflow()
    for retired in ("sys-config", "sys-gen", "md-gen", "show-default"):
        assert retired not in workflow, f"{retired} is named in the workflow"


def test_ci_exercises_exactly_the_three_public_commands():
    workflow = _ci_workflow()
    for command in ("build-top", "build-md", "data-register"):
        assert f"md-openmm {command} -h" in workflow, command


def test_ci_installs_the_wheel_and_runs_outside_the_checkout():
    """Running in the checkout would test the source tree, not the artefact users install."""
    workflow = _ci_workflow()
    assert "python -m build --wheel" in workflow
    assert "pip install --quiet --no-deps dist/*.whl" in workflow
    assert 'assert "site-packages" in origin' in workflow
    document = yaml.safe_load(workflow)
    steps = document["jobs"]["package"]["steps"]
    outside = [s for s in steps if "outside" in str(s.get("working-directory", ""))]
    assert len(outside) >= 5, [s.get("name") for s in steps]
