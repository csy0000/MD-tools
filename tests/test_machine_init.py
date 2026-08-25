"""`md-template init` writes one usable machine.yaml."""
from __future__ import annotations

import yaml


def test_init_creates_the_layout_and_a_valid_machine_file(md_template, tmp_path):
    stack = tmp_path / "MD_STACK"
    result = md_template("init", "--target-dir", str(stack))
    assert result.returncode == 0, result.stderr

    for name in ("envs", "packages", "logs"):
        assert (stack / name).is_dir(), name
    document = yaml.safe_load((stack / "machine.yaml").read_text())

    assert document["schema_version"] == 1
    machine = document["machine"]
    assert machine["hostname"] and machine["architecture"]
    assert machine["logical_cpu_cores"] >= 1
    assert set(document["gpu"]) == {"available", "driver_version", "cuda_version", "devices"}
    assert document["paths"]["md_stack"] == str(stack)
    assert document["installed"] == {"openmm": None}
    assert f"export MD_STACK={stack}" in result.stdout


def test_the_underscore_spelling_is_accepted(md_template, tmp_path):
    stack = tmp_path / "stack2"
    assert md_template("init", "--target_dir", str(stack)).returncode == 0
    assert (stack / "machine.yaml").is_file()


def test_reinitialising_keeps_what_was_installed(md_template, tmp_path):
    """Re-running init re-inspects the hardware; it must not forget the installed engine."""
    stack = tmp_path / "stack3"
    md_template("init", "--target-dir", str(stack))
    path = stack / "machine.yaml"
    document = yaml.safe_load(path.read_text())
    document["installed"]["openmm"] = {"version": "8.6.0", "environment": "/somewhere"}
    path.write_text(yaml.safe_dump(document))

    md_template("init", "--target-dir", str(stack))
    assert yaml.safe_load(path.read_text())["installed"]["openmm"]["version"] == "8.6.0"
