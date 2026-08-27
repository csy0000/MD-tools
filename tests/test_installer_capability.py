"""MD-data readiness: one record shape, said plainly, and written down.

The gaps these close are reporting gaps, not scientific ones. A failed pip install used to return
`{"installed": False, "error": ...}` with no `reasons` -- while the console and `machine.yaml` both
read `reasons`, so the failure was recorded and then displayed as an empty list. And the computed
capability never reached `machine.yaml` at all, so the one durable record of what an environment
can do said nothing about whether contract-managed generation would work.
"""
from __future__ import annotations

import io
import subprocess

import pytest
import yaml

from md_templates.install import openmm as installer
from md_templates.openmm import md_data_contract as MD

REQUIRED = {"repository", "commit", "contract_version", "requirement", "attempted", "command",
            "returncode", "installed", "installed_version", "installed_commit",
            "installed_source", "source_kind", "commit_verified", "contract_support_ready",
            "reasons", "error"}


class _Result:
    """Just enough of `subprocess.CompletedProcess` for the installer to read."""

    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _probe(**fields):
    base = {"import_ok": True, "validator_available": True, "version": "0.2.0",
            "contract_version": MD.MD_DATA_CONTRACT_VERSION, "installed_commit": None,
            "installed_source": None, "source_kind": "unknown"}
    base.update(fields)
    return base


# --- 1. every outcome has the same complete record ----------------------------------------------

@pytest.mark.parametrize("label, returncode, probe", [
    ("verified", 0, _probe(installed_commit=MD.MD_DATA_COMMIT, source_kind="vcs")),
    ("wrong commit", 0, _probe(installed_commit="b" * 40, source_kind="vcs")),
    ("no source commit", 0, _probe(source_kind="local directory",
                                   installed_source="file:///somewhere")),
    ("contract disagrees", 0, _probe(installed_commit=MD.MD_DATA_COMMIT, source_kind="vcs",
                                     contract_version="2.0")),
    ("no validator", 0, _probe(installed_commit=MD.MD_DATA_COMMIT, source_kind="vcs",
                               validator_available=False)),
    ("does not import", 0, {"import_ok": False, "validator_available": False,
                            "error": "ModuleNotFoundError: md_data"}),
    ("pip failed", 1, None),
])
def test_every_outcome_returns_the_same_complete_record(monkeypatch, label, returncode, probe):
    monkeypatch.setattr(installer.subprocess, "run",
                        lambda *a, **k: _Result(returncode, stderr="fatal: could not read "
                                                                   "Username for 'https://github.com'"))
    if probe is not None:
        monkeypatch.setattr(installer, "probe_md_data", lambda prefix: probe)

    record = installer.install_md_data("/nowhere")
    missing = REQUIRED - set(record)
    assert not missing, f"{label}: missing {sorted(missing)}"
    assert record["attempted"] is True
    assert record["commit"] == MD.MD_DATA_COMMIT
    assert record["requirement"] == MD.md_data_requirement()

    if label == "verified":
        assert record["contract_support_ready"] is True
        assert record["commit_verified"] is True
        assert record["reasons"] == []
    else:
        assert record["contract_support_ready"] is False, label
        # The reason has to be HERE, not only in `error`: this is what the console reads.
        assert record["reasons"], f"{label}: contract unavailable with no reason given"
        assert all(isinstance(reason, str) and reason for reason in record["reasons"])
        # `commit_verified` is a narrower fact than readiness and is deliberately independent: a
        # package can be from exactly the pinned commit and still fail on a missing validator or a
        # contract version this repository does not target.
        expected = label in ("contract disagrees", "no validator")
        assert record["commit_verified"] is expected, label


def test_a_pip_failure_carries_an_actionable_reason_not_only_an_error_field(monkeypatch):
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: _Result(
        1, stderr="fatal: could not read Username for 'https://github.com'"))
    record = installer.install_md_data("/nowhere")

    assert record["installed"] is False
    assert record["contract_support_ready"] is False
    assert record["commit_verified"] is False
    assert record["returncode"] == 1
    assert record["reasons"], "a failed install with no reasons is how this got missed"
    assert "pip exit 1" in record["reasons"][0]
    assert "could not read Username" in record["reasons"][0]
    # ...and it tells the user what to do about it.
    assert any(MD.md_data_requirement() in reason for reason in record["reasons"])
    assert "could not read Username" in record["error"]


def test_a_dry_run_does_not_claim_either_outcome(monkeypatch):
    monkeypatch.setattr(installer.subprocess, "run",
                        lambda *a, **k: pytest.fail("a dry run executed pip"))
    record = installer.install_md_data("/nowhere", dry_run=True)
    assert record["attempted"] is False
    assert record["installed"] is None
    # Not False: nothing was installed, so readiness is not a question with an answer yet.
    assert record["contract_support_ready"] is None
    assert "not evaluated" in record["reasons"][0]
    assert record["command"][1:] == ["install", MD.md_data_requirement()]


# --- 2. the console says it plainly -------------------------------------------------------------

def _report(result):
    from md_templates.cli.md_template import _report_environment

    buffer = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(buffer):
        _report_environment(result)
    return buffer.getvalue()


def _environment(md_data, capabilities):
    return {"openmm_version": "8.6.0", "python_version": "3.12.14", "platforms": ["CUDA"],
            "md_data": md_data, "capabilities": capabilities}


def test_the_console_says_ready():
    text = _report(_environment(
        {"commit": MD.MD_DATA_COMMIT, "installed_version": "0.2.0", "source_kind": "vcs",
         "installed_commit": MD.MD_DATA_COMMIT, "attempted": True},
        installer.capability_summary({"contract_support_ready": True, "reasons": []})))
    assert "md-data contract: ready" in text
    assert MD.MD_DATA_COMMIT in text
    assert "UNAVAILABLE" not in text


def test_the_console_says_unavailable_and_why():
    reason = ("the installed md-data records no source commit (installed from local directory), "
              "so it cannot be shown to be the pinned 48628f9a5d3a")
    text = _report(_environment(
        {"commit": MD.MD_DATA_COMMIT, "installed_version": "0.2.0",
         "source_kind": "local directory", "attempted": True},
        installer.capability_summary({"contract_support_ready": False, "reasons": [reason]})))
    assert "md-data contract: UNAVAILABLE" in text
    assert reason in text
    assert "dataset.enabled: true` will fail" in text
    # A reader must not be able to mistake a working OpenMM for a working contract.
    assert "unregistered local simulation is unaffected" in text


def test_the_console_does_not_claim_an_outcome_for_a_dry_run():
    text = _report(_environment({"attempted": False, "commit": MD.MD_DATA_COMMIT},
                                {"md_data_contract_support_ready": False,
                                 "md_data_unavailable_reasons": ["dry run"]}))
    assert "not evaluated (dry run)" in text
    assert "UNAVAILABLE" not in text and "md-data contract: ready" not in text


def test_the_unavailable_banner_is_not_printed_twice():
    """`warnings_for` also produces one; the console must not repeat it as a loose note."""
    check = {"cuda_available": True, "nvidia": {"present": True},
             "capabilities": installer.capability_summary(
                 {"contract_support_ready": False, "reasons": ["md_data does not import"]})}
    notes = installer.warnings_for(check)
    assert any("MD-DATA CONTRACT SUPPORT UNAVAILABLE" in note for note in notes)
    text = _report({**_environment(
        {"commit": MD.MD_DATA_COMMIT, "attempted": True}, check["capabilities"]),
        "warnings": notes})
    assert text.count("md-data contract: UNAVAILABLE") == 1
    assert "note          : MD-DATA CONTRACT SUPPORT UNAVAILABLE" not in text


# --- 3. it survives into machine.yaml -----------------------------------------------------------

@pytest.fixture
def stack(tmp_path):
    from md_templates.install.inspect import initialise

    initialise(tmp_path)
    return tmp_path


@pytest.mark.parametrize("ready, reasons", [
    (False, ["the installed md-data records no source commit, so it cannot be shown to be the "
             "pinned 48628f9a5d3a"]),
    (True, []),
])
def test_the_capability_survives_a_machine_yaml_round_trip(stack, ready, reasons):
    md_data = {**installer.md_data_pin(), "installed": True, "installed_version": "0.2.0",
               "installed_commit": (MD.MD_DATA_COMMIT if ready else None),
               "source_kind": ("vcs" if ready else "local directory"),
               "commit_verified": ready, "contract_support_ready": ready, "reasons": reasons}
    check = {"openmm_version": "8.6.0", "md_data": md_data,
             "capabilities": installer.capability_summary(md_data)}

    installer.record_environment(stack, stack / "envs" / "openmm-8.6.0", check)
    reloaded = yaml.safe_load((stack / "machine.yaml").read_text())["installed"]["openmm"]

    assert reloaded["capabilities"]["md_data_contract_support_ready"] is ready
    assert reloaded["capabilities"]["md_data_unavailable_reasons"] == reasons
    assert reloaded["md_data"]["commit"] == MD.MD_DATA_COMMIT
    assert reloaded["md_data"]["contract_support_ready"] is ready
    assert reloaded["md_data"]["commit_verified"] is ready
    if not ready:
        assert "no source commit" in reloaded["capabilities"]["md_data_unavailable_reasons"][0]


def test_machine_yaml_holds_only_plain_data(stack):
    """A machine.yaml is read months later by something that is not this code."""
    md_data = {**installer.md_data_pin(), "contract_support_ready": False,
               "reasons": ["boom"],
               # The kind of thing that must never be stored verbatim.
               "probe": {"result": subprocess.CompletedProcess(args=["x"], returncode=1)},
               "raised": ValueError("nope")}
    installer.record_environment(stack, stack / "env", {"md_data": md_data,
                                                        "capabilities": installer
                                                        .capability_summary(md_data)})
    text = (stack / "machine.yaml").read_text()
    assert "!!python" not in text, "a non-plain object was serialised into machine.yaml"
    reloaded = yaml.safe_load(text)["installed"]["openmm"]["md_data"]
    assert isinstance(reloaded["raised"], str) and "nope" in reloaded["raised"]
    assert isinstance(reloaded["probe"]["result"], str)


def test_validation_records_the_same_shape_as_installation(stack, monkeypatch):
    """An environment validated later must not need a machine.yaml reader to special-case it."""
    md_data = {**installer.md_data_pin(), "contract_support_ready": False,
               "commit_verified": False, "reasons": ["md_data does not import"],
               "installed_version": None, "installed_commit": None, "source_kind": None}
    monkeypatch.setattr(installer, "verify_environment",
                        lambda prefix, version="": {"openmm_version": "8.6.0", "warnings": []})
    monkeypatch.setattr(installer, "verify_md_data", lambda prefix: md_data)

    result = installer.validate_existing(stack, stack / "envs" / "openmm-8.6.0")
    assert result["capabilities"]["md_data_contract_support_ready"] is False
    assert result["md_data"]["reasons"] == ["md_data does not import"]

    reloaded = yaml.safe_load((stack / "machine.yaml").read_text())["installed"]["openmm"]
    assert set(reloaded["capabilities"]) == {"openmm_runtime_ready",
                                             "md_data_contract_support_ready",
                                             "md_data_unavailable_reasons"}
    assert reloaded["md_data"]["commit"] == MD.MD_DATA_COMMIT
    # ...and validation surfaces the same banner installation does.
    assert any("MD-DATA CONTRACT SUPPORT UNAVAILABLE" in note for note in result["warnings"])


# --- 4. the REAL command path, and one canonical record shape -----------------------------------

def _dry_run_cli(stack, monkeypatch):
    """`md-template install -e openmm -ev 8.6.0 --dry-run`, through main().

    Only the package-manager lookup and the CUDA-ceiling probe are stubbed -- both shell out, and
    a dry run must not. `subprocess.run` is made to raise, so any attempt to execute anything at
    all fails the test rather than silently succeeding.
    """
    import contextlib

    from md_templates.cli import md_template as cli

    def executed(*args, **kwargs):
        raise AssertionError("a dry run executed a subprocess")

    monkeypatch.setattr(installer, "find_package_manager",
                        lambda: ("micromamba", "/usr/bin/micromamba"))
    monkeypatch.setattr(installer, "driver_cuda_ceiling", lambda: "13.0")
    monkeypatch.setattr(installer.subprocess, "run", executed)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(["install", "-e", "openmm", "-ev", "8.6.0", "--dry-run",
                         "--target-dir", str(stack)])
    return code, buffer.getvalue()


def test_the_real_dry_run_command_reports_md_data_as_not_evaluated(stack, monkeypatch):
    """The branch existed and was unreachable: install_openmm returned before building the record.

    Exercised through `main()` rather than by calling the reporter, because "the code can print
    this" and "the command prints this" are different claims and only the second one matters.
    """
    code, text = _dry_run_cli(stack, monkeypatch)
    assert code == 0
    assert "md-data contract: not evaluated (dry run)" in text
    # It must claim neither outcome.
    assert "ready" not in text
    assert "UNAVAILABLE" not in text


def test_the_real_dry_run_installs_validates_and_records_nothing(stack, monkeypatch):
    _dry_run_cli(stack, monkeypatch)
    machine = yaml.safe_load((stack / "machine.yaml").read_text())
    assert not (machine.get("installed") or {}).get("openmm"), \
        "a dry run wrote an installed-environment record"
    assert not (stack / "envs" / "openmm-8.6.0").exists(), "a dry run created the environment"


def test_the_dry_run_result_carries_the_structured_record(stack, monkeypatch):
    monkeypatch.setattr(installer, "find_package_manager",
                        lambda: ("micromamba", "/usr/bin/micromamba"))
    monkeypatch.setattr(installer, "driver_cuda_ceiling", lambda: "13.0")
    monkeypatch.setattr(installer.subprocess, "run",
                        lambda *a, **k: pytest.fail("a dry run executed a subprocess"))

    result = installer.install_openmm(stack, "8.6.0", dry_run=True)
    assert result["dry_run"] is True
    assert set(result["md_data"]) == set(installer._md_data_record(installer.md_data_pin()))
    assert result["md_data"]["attempted"] is False
    assert result["md_data"]["contract_support_ready"] is None
    assert result["capabilities"]["md_data_contract_support_ready"] is False
    assert "not evaluated" in result["md_data"]["reasons"][0]


def _installed_record(monkeypatch, probe):
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: _Result(0))
    monkeypatch.setattr(installer, "probe_md_data", lambda prefix: probe)
    return installer.install_md_data("/nowhere")


def test_installation_and_validation_records_have_identical_key_sets(monkeypatch):
    """One shape, from one definition. A key on only one of them is a reader's problem later."""
    probe = _probe(installed_commit=MD.MD_DATA_COMMIT, source_kind="vcs")
    installed = _installed_record(monkeypatch, probe)
    monkeypatch.setattr(installer, "probe_md_data", lambda prefix: probe)
    validated = installer.verify_md_data("/nowhere")

    canonical = set(installer._md_data_record(installer.md_data_pin()))
    assert set(installed) == canonical
    assert set(validated) == canonical
    assert set(installed) == set(validated)


@pytest.mark.parametrize("probe, ready", [
    (_probe(installed_commit=MD.MD_DATA_COMMIT, source_kind="vcs"), True),
    (_probe(source_kind="local directory", installed_source="file:///x"), False),
])
def test_validation_is_evaluated_and_is_not_a_dry_run(monkeypatch, probe, ready):
    """`attempted: null` means "not a pip operation", not "not evaluated"."""
    monkeypatch.setattr(installer, "probe_md_data", lambda prefix: probe)
    record = installer.verify_md_data("/nowhere")

    assert record["attempted"] is None, "false is reserved for a dry run"
    assert record["command"] is None and record["returncode"] is None
    assert record["contract_support_ready"] is ready
    assert record["installed"] is True          # observable: it imported
    assert (record["reasons"] == []) is ready

    text = _report(_environment(record, installer.capability_summary(record)))
    assert "not evaluated" not in text, "validation must not look like a dry run"
    assert ("md-data contract: ready" in text) is ready
    assert ("md-data contract: UNAVAILABLE" in text) is (not ready)


def test_a_validation_record_round_trips_through_machine_yaml(stack, monkeypatch):
    reason_probe = _probe(source_kind="local directory", installed_source="file:///x")
    monkeypatch.setattr(installer, "verify_environment",
                        lambda prefix, version="": {"openmm_version": "8.6.0", "warnings": []})
    monkeypatch.setattr(installer, "probe_md_data", lambda prefix: reason_probe)

    result = installer.validate_existing(stack, stack / "envs" / "openmm-8.6.0")
    reloaded = yaml.safe_load((stack / "machine.yaml").read_text())["installed"]["openmm"]
    md_data = reloaded["md_data"]

    assert set(md_data) == set(installer._md_data_record(installer.md_data_pin()))
    assert md_data["attempted"] is None
    assert md_data["command"] is None and md_data["returncode"] is None
    assert md_data["contract_support_ready"] is False
    assert md_data["reasons"] and "no source commit" in md_data["reasons"][0]
    assert md_data["commit"] == MD.MD_DATA_COMMIT
    assert reloaded["capabilities"]["md_data_contract_support_ready"] is False
    assert reloaded["capabilities"]["md_data_unavailable_reasons"] == md_data["reasons"]
    # The persisted record is what the reporter would speak from, and it is not a dry run.
    assert "not evaluated" not in _report(_environment(md_data, reloaded["capabilities"]))
