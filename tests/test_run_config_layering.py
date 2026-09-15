"""A shared input plus a narrow per-run override, resolving to one authoritative document.

WHY THIS EXISTS, from the data rather than from design. The three ALA-explicit REST2 reference
runs differ in their `.in` files by exactly one line each:

    22c22
    <   random_seed               = 700501,
    >   random_seed               = 700502,

Nothing else. So an input genuinely is not per-repeat -- EXCEPT the seed, which `in_file_text`
currently writes into every `.in`. Sharing `input/` at the dataset root therefore requires the
seed to leave the input and live in a per-run `run.config`, and the two to resolve together.

THE OVERRIDE IS DELIBERATELY ONE KEY WIDE. `derive_seed` hashes the base seed with each stage and
replica name, so every stream in a run descends from that number -- it is the whole reason two
repeats differ. Anything else belongs in the shared input, where every repeat reads the same
value. A file that could set anything would be a second configuration authority, and
`md_tools.build.md` exists to be the only one.

PLATFORM_POLICY_EXEMPTION: configuration resolution only. No Context, no device, no simulation.
"""
from __future__ import annotations

import pytest

from md_tools.build.md import RUN_CONFIG_ALLOWED, load_run_config, resolve_md_config
from md_tools.build.strict import ConfigError


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# -- what the override may carry ----------------------------------------------------------------

def test_only_the_seed_may_be_set_per_run():
    """The allow-list is the contract; widening it is what this test exists to notice."""
    assert RUN_CONFIG_ALLOWED == {"dynamics": frozenset({"seed"})}


def test_the_seed_overrides_the_shared_input(tmp_path):
    shared = _write(tmp_path, "REST2.config", "protocol: REST2\n")
    run = _write(tmp_path, "run.config", "dynamics:\n  seed: 700502\n")

    assert resolve_md_config(shared)["dynamics"]["seed"] == 1          # the schema default
    assert resolve_md_config(shared, run_config=run)["dynamics"]["seed"] == 700502


def test_two_runs_of_one_shared_input_differ_only_in_the_seed(tmp_path):
    """The reference data's shape, reproduced: one input, two runs, one differing value."""
    shared = _write(tmp_path, "REST2.config", "protocol: REST2\n")
    first = resolve_md_config(shared, run_config=_write(
        tmp_path, "a.config", "dynamics:\n  seed: 700501\n"))
    second = resolve_md_config(shared, run_config=_write(
        tmp_path, "b.config", "dynamics:\n  seed: 700502\n"))

    assert first["dynamics"]["seed"] != second["dynamics"]["seed"]
    first["dynamics"]["seed"] = second["dynamics"]["seed"] = 0
    assert first == second, "the seed must be the ONLY difference between two repeats"


def test_an_absent_run_config_leaves_the_default(tmp_path):
    """A run generated before run.config existed must still resolve."""
    shared = _write(tmp_path, "REST2.config", "protocol: REST2\n")
    assert resolve_md_config(shared, run_config=None)["dynamics"]["seed"] == 1
    assert resolve_md_config(shared, run_config=tmp_path / "absent.config"
                             )["dynamics"]["seed"] == 1
    assert load_run_config(None) == {}
    assert load_run_config(tmp_path / "absent.config") == {}


# -- what it refuses ----------------------------------------------------------------------------

def test_a_section_that_is_not_dynamics_is_refused_by_name(tmp_path):
    run = _write(tmp_path, "run.config", "stages:\n  production_steps: 10\n")
    with pytest.raises(ConfigError, match="may not be set per run"):
        load_run_config(run)


def test_a_dynamics_key_that_is_not_the_seed_is_refused_by_name(tmp_path):
    """Two files that can both set a timestep is two answers waiting to disagree."""
    run = _write(tmp_path, "run.config", "dynamics:\n  seed: 5\n  temperature_K: 310.0\n")
    with pytest.raises(ConfigError, match="temperature_K.*may not be set per run"):
        load_run_config(run)


def test_the_refusal_says_what_may_be_set_instead(tmp_path):
    run = _write(tmp_path, "run.config", "reporting:\n  info_printout: 10\n")
    with pytest.raises(ConfigError) as failure:
        load_run_config(run)
    assert "dynamics.seed" in str(failure.value)


def test_a_non_mapping_document_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_run_config(_write(tmp_path, "run.config", "- a\n- b\n"))


def test_a_non_mapping_section_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_run_config(_write(tmp_path, "run.config", "dynamics: 700501\n"))


# -- the override must not disturb protocol-dependent defaults ----------------------------------

def test_layering_does_not_change_which_reporting_intervals_ais_considers_stated(tmp_path):
    """`_apply_ais_reporting_defaults` reads `stated`, so the merge must precede it.

    A bare AIS document resolves its unstated reporting intervals to
    `observation_interval_steps`, because the cMD defaults of 1000/10000 do not divide a short
    switching path. Merging the seed after `stated` was taken would leave that untouched here but
    is the kind of ordering bug that surfaces as a refused run much later.
    """
    shared = _write(tmp_path, "AIS.config",
                    "protocol: AIS\nais_source:\n  trajectory: source.nc\n")
    run = _write(tmp_path, "run.config", "dynamics:\n  seed: 99\n")

    without = resolve_md_config(shared)
    with_seed = resolve_md_config(shared, run_config=run)

    assert with_seed["dynamics"]["seed"] == 99
    for key in ("crd_printout_solute", "info_printout", "checkpoint_printout"):
        assert with_seed["reporting"][key] == without["reporting"][key], key


def test_a_seed_stated_per_run_counts_as_stated(tmp_path):
    """It is a value the user wrote, merely in the other file."""
    shared = _write(tmp_path, "cMD.config", "protocol: cMD\n")
    resolved = resolve_md_config(shared, run_config=_write(
        tmp_path, "run.config", "dynamics:\n  seed: 4242\n"))
    assert resolved["dynamics"]["seed"] == 4242


def test_the_shared_input_still_resolves_on_its_own(tmp_path):
    """Layering is additive: nothing that worked before may now require a run.config."""
    for body in ("protocol: cMD\n",
                 "protocol: REST2\n",
                 "protocol: AIS\nais_source:\n  trajectory: s.nc\n"):
        resolved = resolve_md_config(_write(tmp_path, "c.config", body))
        assert resolved["dynamics"]["seed"] == 1
