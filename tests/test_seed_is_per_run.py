"""The seed leaves the shared input and lives in the per-run `run.config`.

WHY, from the reference data rather than from design. `input/` sits at the dataset root and every
repeat of a method reads the same files. The three ALA-explicit REST2 runs' inputs differ by
exactly one line each:

    <   random_seed               = 700501,
    >   random_seed               = 700502,

Nothing else. So an input genuinely is not per-repeat EXCEPT for the seed -- and `derive_seed`
hashes that one number with each stage and replica name, so every stream in a run descends from
it. Two runs sharing an input and a seed would be bit-identical rather than repeats.

EMISSION AND ACCEPTANCE ARE DIFFERENT QUESTIONS, and this file pins both. A generated `.in` no
longer WRITES `random_seed`; `SECTION_KEYS` still ACCEPTS it, because nine migrated reference runs
carry it in their inputs and `md-run` must keep reading them. Conflating the two would make every
existing input unparseable.

PLATFORM_POLICY_EXEMPTION: generation, parsing and resolution only. No Context, no device.
"""
from __future__ import annotations

import pytest

from md_tools.build.md import _NOT_IN_INPUT, in_file_text, resolve_md_config
from md_tools.run.inputs import SECTION_KEYS, parse_run_input

BODIES = {
    "cMD": "protocol: cMD\n",
    "REST2": "protocol: REST2\n",
    "AIS": "protocol: AIS\nais_source:\n  trajectory: source.nc\n",
    "umbrella": ("protocol: umbrella\numbrella:\n  file: windows.yaml\n"
                 "collective_variables:\n  file: cv.yaml\n  interval_steps: 1000\n"),
}


def _generated(tmp_path, protocol):
    config = tmp_path / f"{protocol}.config"
    config.write_text(BODIES[protocol], encoding="utf-8")
    resolved = resolve_md_config(config)
    text = in_file_text(resolved, heading=protocol)
    path = tmp_path / f"{protocol}.in"
    path.write_text(text, encoding="utf-8")
    return path, text


# -- emission -----------------------------------------------------------------------------------

def test_the_exclusion_is_declared_by_target_not_by_key():
    """`random_seed` is accepted in &cntrl AND &AIS, so keying by name would miss one."""
    assert _NOT_IN_INPUT == frozenset({"dynamics.seed"})


@pytest.mark.parametrize("protocol", sorted(BODIES))
def test_a_generated_input_does_not_write_the_seed(tmp_path, protocol):
    _path, text = _generated(tmp_path, protocol)
    assert "random_seed" not in text, f"{protocol} still emits the seed into a shared input"


def test_the_seed_is_excluded_from_the_ais_block_too(tmp_path):
    """It is accepted in &AIS deliberately, so an AIS input reads as one block."""
    assert SECTION_KEYS["AIS"]["random_seed"] == "dynamics.seed"
    _path, text = _generated(tmp_path, "AIS")
    assert "&AIS" in text and "random_seed" not in text


# -- acceptance ---------------------------------------------------------------------------------

def test_an_input_that_carries_a_seed_still_parses():
    """Nine migrated reference runs carry `random_seed`; md-run must keep reading them."""
    assert SECTION_KEYS["cntrl"]["random_seed"] == "dynamics.seed"


def test_an_existing_input_with_a_seed_resolves_to_that_seed(tmp_path):
    path = tmp_path / "legacy.in"
    path.write_text("! a run generated before the seed moved\n"
                    "&cntrl\n  protocol = cMD,\n  random_seed = 700501,\n/\n", encoding="utf-8")
    assert parse_run_input(path).resolved["dynamics"]["seed"] == 700501


# -- layering -----------------------------------------------------------------------------------

@pytest.mark.parametrize("protocol", sorted(BODIES))
def test_a_generated_input_resolves_to_the_default_seed_alone(tmp_path, protocol):
    path, _text = _generated(tmp_path, protocol)
    assert parse_run_input(path).resolved["dynamics"]["seed"] == 1


@pytest.mark.parametrize("protocol", sorted(BODIES))
def test_a_run_config_beside_it_supplies_the_seed(tmp_path, protocol):
    path, _text = _generated(tmp_path, protocol)
    run_config = tmp_path / "run.config"
    run_config.write_text("dynamics:\n  seed: 700502\n", encoding="utf-8")
    assert parse_run_input(path, run_config=run_config).resolved["dynamics"]["seed"] == 700502


def test_two_runs_of_one_shared_input_differ_only_in_the_seed(tmp_path):
    """The reference data's shape: one input, two runs, one differing value."""
    path, _text = _generated(tmp_path, "REST2")
    first_config = tmp_path / "a.config"
    first_config.write_text("dynamics:\n  seed: 700501\n", encoding="utf-8")
    second_config = tmp_path / "b.config"
    second_config.write_text("dynamics:\n  seed: 700502\n", encoding="utf-8")

    first = parse_run_input(path, run_config=first_config).resolved
    second = parse_run_input(path, run_config=second_config).resolved
    assert first["dynamics"]["seed"] != second["dynamics"]["seed"]
    first["dynamics"]["seed"] = second["dynamics"]["seed"] = 0
    assert first == second, "the seed must be the ONLY difference between two repeats"


def test_an_absent_run_config_is_an_empty_override(tmp_path):
    """md-run passes `-odir/run.config` always; most runs will not have one."""
    path, _text = _generated(tmp_path, "cMD")
    assert parse_run_input(path, run_config=tmp_path / "absent.config"
                           ).resolved["dynamics"]["seed"] == 1


def test_md_run_passes_the_override_from_its_output_directory():
    """Located in `md-run`, not discovered by the parser.

    A parser that went looking for a file beside its input would let a run silently acquire the
    one setting whose silent acquisition makes two repeats bit-identical.
    """
    from pathlib import Path

    source = Path("src/md_tools/run/main.py").read_text(encoding="utf-8")
    assert 'run_config=Path(args.out_dir) / "run.config"' in source
