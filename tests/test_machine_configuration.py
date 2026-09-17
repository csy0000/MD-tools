"""The machine configuration: absent, valid, and every way of being wrong.

The distinction this file exists for is between an ABSENT configuration and an INVALID one.

Absent is legitimate. Running a simulation needs neither an identity nor a storage root, so a
machine where nobody has written a user configuration resolves to the built-in defaults -- CUDA,
mixed precision, local-rank placement.

Invalid is not. A file that exists and says something wrong is a mistake a person made in a file
they edited, and the previous behaviour answered it with those same defaults: bad YAML, a
duplicate key, an unknown field or an invalid platform all became CUDA/mixed/local_rank. The
machine then ran on settings nobody chose, and the file that was supposed to say otherwise was
never mentioned again.

Every invalid case is also checked through the real command, with an assertion that the filesystem
is untouched -- because the point is not only that it refuses but that it refuses BEFORE writing.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.registry.errors import RegistrationError
from md_tools.registry.userconfig import machine_openmm_settings, resolve_machine_openmm

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

VALID_USER = {"person_id": "someone", "name": "Some One", "orcid": None, "affiliation": None}


def _config(tmp_path: Path, document) -> Path:
    path = tmp_path / "user.config"
    path.write_text(document if isinstance(document, str) else yaml.safe_dump(document),
                    encoding="utf-8")
    return path


# --- what is accepted ---------------------------------------------------------------------------

def test_no_configuration_file_at_all(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("MD_TOOLS_CONFIG", raising=False)
    resolved = machine_openmm_settings()
    assert (resolved["platform"], resolved["precision"], resolved["device_policy"]) == \
        ("CUDA", "mixed", "local_rank")
    assert resolved["origin"] == "built-in default"
    assert resolved["config_path"] is None


def test_a_legacy_configuration_predating_the_openmm_block(tmp_path):
    """Written before `machine.openmm` existed. Still valid; it simply says nothing about OpenMM."""
    path = _config(tmp_path, {"schema_version": "1.0", "user": VALID_USER,
                              "machine": {"md_data": "/tmp"}})
    resolved = machine_openmm_settings(path)
    assert resolved["platform"] == "CUDA"
    assert resolved["origin"] == "built-in default"
    assert resolved["config_path"] == str(path)


@pytest.mark.parametrize("platform", ["CUDA", "CPU"])
def test_a_valid_machine_block_is_honoured_and_its_origin_recorded(platform, tmp_path):
    path = _config(tmp_path, {
        "schema_version": "1.0", "user": VALID_USER,
        "machine": {"md_data": "/tmp",
                    "openmm": {"platform": platform, "precision": "mixed",
                               "device_policy": "local_rank"}}})
    resolved = machine_openmm_settings(path)
    assert resolved["platform"] == platform
    assert resolved["origin"] == "machine.openmm"


@pytest.mark.parametrize("policy", ["local_rank", "openmm"])
def test_both_device_policies_are_accepted_and_do_something(policy, tmp_path):
    """Accepted AND implemented. `openmm` used to be validated and never consulted."""
    from md_tools.openmm.platform_policy import device_index_for

    path = _config(tmp_path, {
        "schema_version": "1.0", "user": VALID_USER,
        "machine": {"openmm": {"device_policy": policy}}})
    assert machine_openmm_settings(path)["device_policy"] == policy

    chosen = device_index_for(policy=policy, rank=1, size=2, devices=["0", "1"])
    assert (chosen is None) == (policy == "openmm"), policy


# --- what is refused ----------------------------------------------------------------------------

INVALID = {
    "malformed YAML": ("machine:\n  openmm:\n   platform: CUDA\n  bad indent here\n", "YAML"),
    "a non-mapping document": ("- one\n- two\n", "mapping"),
    "a non-mapping user block": ("schema_version: \"1.0\"\nuser: [a, b]\n", "mapping"),
    "a non-mapping machine block": ("schema_version: \"1.0\"\nmachine: 3\n", "mapping"),
    "an unsupported schema version": ("schema_version: \"9.9\"\nuser: {person_id: t, name: T}\n",
                                      "schema_version"),
    "a duplicate machine section": ("machine:\n  md_data: /a\nmachine:\n  md_data: /b\n",
                                    "machine"),
    "a duplicate openmm section": ("machine:\n  openmm:\n    platform: CUDA\n"
                                   "  openmm:\n    platform: CPU\n", "openmm"),
    "a duplicate platform key": ("machine:\n  openmm:\n    platform: CUDA\n    platform: CPU\n",
                                 "platform"),
    "an unknown openmm field": ("machine:\n  openmm:\n    accelerator: fast\n", "accelerator"),
    "an unsupported platform": ("machine:\n  openmm:\n    platform: OpenCL\n", "platform"),
    "an unsupported precision": ("machine:\n  openmm:\n    precision: quadruple\n", "precision"),
    "an unsupported device policy": ("machine:\n  openmm:\n    device_policy: whatever\n",
                                     "device_policy"),
}


@pytest.mark.parametrize("case", sorted(INVALID))
def test_an_invalid_configuration_is_refused_and_never_becomes_the_defaults(case, tmp_path):
    document, fragment = INVALID[case]
    path = _config(tmp_path, document)
    with pytest.raises(RegistrationError) as refusal:
        machine_openmm_settings(path)
    message = str(refusal.value)
    assert fragment.lower() in message.lower(), (case, message)
    assert str(path) in message, f"{case}: the refusal does not name the file"


def test_a_missing_explicit_path_is_a_broken_reference(tmp_path):
    with pytest.raises(RegistrationError, match="does not exist"):
        machine_openmm_settings(tmp_path / "never-written.config")


def test_a_missing_environment_path_is_a_broken_reference(tmp_path, monkeypatch):
    """`MD_TOOLS_CONFIG=/gone` is not the same as leaving it unset: somebody meant that path."""
    monkeypatch.setenv("MD_TOOLS_CONFIG", str(tmp_path / "gone.config"))
    with pytest.raises(RegistrationError) as refusal:
        machine_openmm_settings()
    assert "MD_TOOLS_CONFIG" in str(refusal.value)


def test_the_resolver_itself_refuses_an_unknown_field():
    """Directly, so the rule is not only enforced by the file reader around it."""
    with pytest.raises(RegistrationError, match="turbo"):
        resolve_machine_openmm({"machine": {"openmm": {"turbo": True}}})


# --- and none of it writes anything --------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("case", ["a duplicate platform key", "an unsupported platform",
                                  "malformed YAML"])
def test_an_invalid_configuration_stops_the_command_before_any_output(case, tmp_path):
    """Through the real command, with the filesystem checked afterwards.

    A refusal that has already created `-odir` and written `resolved.config` leaves a directory
    that reads as a run which happened.
    """
    document, _ = INVALID[case]
    broken = _config(tmp_path, document)

    # A dataset root with its built System: `build-md` validates the chain against it and refuses
    # a root without one. The stand-in System is enough -- the subject is the machine configuration.
    from .conftest import make_dataset_root

    work = make_dataset_root(tmp_path / "work", solvent="explicit")
    (work / "cMD.config").write_text(yaml.safe_dump({"protocol": "cMD", "solvent": "explicit"}),
                                     encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(work / "cMD-run1"),
                                      "--config", str(work / "cMD.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    destination = tmp_path / "never"
    done = subprocess.run(
        CLI + ["md-run", "-i", "../input/cMD.in", "-p", "../build/built.pdb",
               "-s", "../build/built.xml",
               "-odir", str(destination)],
        cwd=work / "cMD-run1", capture_output=True, text=True, timeout=600,
        env=dict(os.environ, MD_TOOLS_CONFIG=str(broken)))

    assert done.returncode != 0, done.stdout
    # REFUSED FOR THE CONFIGURATION. The inputs are a real built System now, so nothing else about
    # this invocation is wrong; the fake `<System/>` it used to pass could have been the reason.
    assert str(broken) in done.stdout + done.stderr, done.stdout + done.stderr
    assert not destination.exists(), sorted(p.name for p in destination.iterdir())
