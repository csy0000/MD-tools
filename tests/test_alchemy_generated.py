"""S4/X1: a generated alchemical directory, and the two surfaces that run a window from it.

`md-openmm build-md` writes the directory; `python w000.py` and `md-openmm md-run -i
../input/alchemical.in -odir . --window w000` both reach `window_main`, which is the only place a
window is dispatched. What is asserted here is that they reach the SAME dispatch with the same
settings, and that what each refuses it refuses BY NAME before anything exists.

THE PROPERTY THAT MATTERS is the one a generated directory makes a claim about: it runs the
Hamiltonian it was generated from. The plan is copied in content-addressed, the leg's System is
built once and its digest recorded, and the runtime rebuilds and compares rather than trusting
either side. A directory that referred to a plan elsewhere would break silently where that path
exists and holds something else.

PLATFORM_POLICY_EXEMPTION: every simulating test here passes an explicit `--cpu`. This exercises
the two surfaces and is NOT CUDA evidence (see the S4 acceptance matrix). The caveat is sharper
than the usual one: `LocalEnergyMinimizer.minimize` is not deterministic on CUDA for this System
(S0, 2026-09-29 -- 16 kJ/mol spread in mixed precision, 24 in double, from bit-identical starting
energies), so these tests agreeing twice on the CPU says nothing about whether a window on a card
repeats. What they assert is the WIRING -- which dispatch is reached, what is refused, what is
written -- and none of it depends on the trajectory.
"""
from __future__ import annotations

import json
import os
import textwrap

import pytest
import yaml

os.environ.setdefault("PYMBAR_DISABLE_JAX", "true")
openmm = pytest.importorskip("openmm")
from openmm import app  # noqa: E402

from tests import alchemy_fixtures as af  # noqa: E402
from md_tools.build.md import build_scripts  # noqa: E402
from md_tools.run.inputs import parse_run_input  # noqa: E402
from md_tools.run.main import md_run_main  # noqa: E402

NAMES = ["lambda_bonded", "lambda_electrostatics", "lambda_sterics"]

#: Four windows of 2 ps. Small enough to run twice on the CPU inside a fast lane, and the numbers
#: are never read: what is under test is the wiring, not a free energy.
CONFIG = """
    protocol: alchemical
    alchemical:
      plan: plan
      lambda_path: linear
      number_of_windows: 4
      window_steps: 2000
      equilibration_steps: 200
      report_interval_steps: 100
      checkpoint_interval_steps: 1000
      minimize_iterations: 50
    dynamics:
      timestep_fs: 1.0
      temperature_K: 300.0
      seed: 7
"""


@pytest.fixture(scope="module")
def plan_directory(tmp_path_factory):
    """S2's ethane -> chloroethane hybrid plan in vacuum, written out as `build-md` reads it."""
    from md_tools.alchemy.topology import build_topology_plan

    a, b = af.package(af.ETHANE), af.package(af.CHLOROETHANE)
    plan = build_topology_plan(a, b, af.core_map(a, b),
                               af.vacuum_environment(a, constraints=app.HBonds), mode="hybrid")
    directory = tmp_path_factory.mktemp("plan-source") / "plan"
    plan.write(directory)
    return directory, plan


def _generate(tmp_path, plan_directory, body=CONFIG):
    """`build-md` into `<system>/alchemical-run1`, from a config beside the plan."""
    directory, plan = plan_directory
    config = tmp_path / "alchemical.config"
    config.write_text(textwrap.dedent(body).replace("plan: plan", f"plan: {directory}"),
                      encoding="utf-8")
    run = tmp_path / "ethane-chloroethane" / "alchemical-run1"
    build_scripts(config_path=config, out_dir=run, echo=False)
    return run, plan


def _run_md_run(run, *args):
    """`md-openmm md-run` from inside the run directory, as `run.sh` invokes it."""
    previous = os.getcwd()
    os.chdir(run)
    try:
        return md_run_main(["-i", "../input/alchemical.in", "-odir", ".", *args])
    finally:
        os.chdir(previous)


# --- what the directory holds -----------------------------------------------------------------

def test_a_generated_directory_carries_its_plan_and_its_leg(tmp_path, plan_directory):
    run, plan = _generate(tmp_path, plan_directory)
    copied = run / f"plan.{plan.sha256[:12]}"
    # CONTENT-ADDRESSED AND COPIED IN. Not referred to: a run directory that named a plan outside
    # itself would break the moment it was copied to a cluster, and break silently where that
    # path exists and holds a different plan.
    assert (copied / "plan.json").is_file()
    assert json.loads((run / "leg" / "leg.json").read_text())["plan_sha256"] == plan.sha256
    assert (run / "leg" / "system.xml").is_file() and (run / "leg" / "topology.pdb").is_file()
    assert sorted(p.name for p in run.glob("w*.py")) == ["w000.py", "w001.py", "w002.py",
                                                         "w003.py"]
    # The declaration names the COPY, so the runtime never resolves the source path again.
    resolved = yaml.safe_load((run / "resolved.config").read_text())
    assert resolved["alchemical"]["plan"] == copied.name


def test_a_window_script_is_an_entry_point(tmp_path, plan_directory):
    """Two lines of body. No argparse, no OpenMM, no Hamiltonian built in the generated file."""
    run, _ = _generate(tmp_path, plan_directory)
    text = (run / "w002.py").read_text()
    # Everything after the module docstring IS the body.
    code = [line for line in text.split('"""')[-1].splitlines() if line.strip()]
    assert code == ["from md_tools.alchemy import run_generated_window",
                    'raise SystemExit(run_generated_window(__file__, "w002"))']
    for forbidden in ("import openmm", "argparse", "def ", "class "):
        assert forbidden not in text


def test_the_input_resolves_back_to_the_resolved_config_beside_it(tmp_path, plan_directory):
    """The round-trip every generated `.in` owes: a test, not a convention.

    `run.config` is part of it. The `.in` is SHARED by every repeat of this transformation and so
    carries no seed; the seed is this run's, and lives beside the run.
    """
    run, _ = _generate(tmp_path, plan_directory)
    generated = yaml.safe_load((run / "resolved.config").read_text())
    from_input = parse_run_input(run / ".." / "input" / "alchemical.in",
                                 run_config=run / "run.config").resolved
    assert from_input == generated


def test_run_sh_names_every_window_and_no_system(tmp_path, plan_directory):
    """A ladder takes no -p and no -s: its System is the leg, built once and digest-checked."""
    run, _ = _generate(tmp_path, plan_directory)
    text = (run / "run.sh").read_text()
    assert 'WINDOWS="${WINDOWS:-w000 w001 w002 w003}"' in text
    # The one command it runs. Checked as the whole line, not as a substring of the file: the
    # comments above it name `-p` and `-s` precisely to say that they are not passed.
    command = [line.strip() for line in text.splitlines()
               if line.strip().startswith("md-openmm md-run")]
    assert command == ['md-openmm md-run -i ../input/alchemical.in -odir . '
                       '--window "${window}" "$@"']
    assert "built.xml" not in text and "built.pdb" not in text


# --- the two surfaces, on the same window -----------------------------------------------------

def test_both_surfaces_check_the_same_window_and_create_nothing(tmp_path, plan_directory):
    """`--check` creates nothing, not even -odir -- through either entrance."""
    run, _ = _generate(tmp_path, plan_directory)
    before = sorted(p.name for p in run.iterdir())

    assert _run_md_run(run, "--window", "w000", "--cpu", "--check") == 0
    assert sorted(p.name for p in run.iterdir()) == before
    assert not (run / "leg" / "windows").exists()

    from md_tools.alchemy.generated import run_generated_window
    assert run_generated_window(run / "w000.py", "w000", ["--cpu", "--check"]) == 0
    assert sorted(p.name for p in run.iterdir()) == before
    assert not (run / "leg" / "windows").exists()


def test_a_window_runs_from_either_surface_and_is_then_skipped(tmp_path, plan_directory, capsys):
    """The acceptance criterion: a window runs end to end from a generated directory.

    Then the SAME window through the OTHER surface is skipped rather than rerun -- which is the
    evidence that the two entrances reach one dispatch and one completion record, not two.
    """
    from md_tools.alchemy.generated import run_generated_window
    from md_tools.alchemy.windows import window_paths

    run, _ = _generate(tmp_path, plan_directory)
    assert run_generated_window(run / "w000.py", "w000", ["--cpu"]) == 0
    assert "w000: fresh" in capsys.readouterr().out
    completion = json.loads(
        window_paths(run / "leg" / "windows" / "r1", "w000")["completion"].read_text())
    assert completion["window_id"] == "w000" and completion["rows"] > 0

    # THE OTHER surface, the same window: skipped by its own completion record, not rerun.
    assert _run_md_run(run, "--window", "w000", "--cpu") == 0
    assert "w000: verified-complete" in capsys.readouterr().out
    assert json.loads(window_paths(run / "leg" / "windows" / "r1",
                                   "w000")["completion"].read_text()) == completion

    # And a second window, from the other surface again, into the same leg and the same repeat.
    assert _run_md_run(run, "--window", "w001", "--cpu") == 0
    assert "w001: fresh" in capsys.readouterr().out


# --- what each surface refuses, by name -------------------------------------------------------

@pytest.mark.parametrize("flag", ["-p", "-s", "-groupfile"])
def test_a_ladder_refuses_a_second_system_by_name(tmp_path, plan_directory, capsys, flag):
    """One Hamiltonian for the windows: the leg's. A second named here would be a second answer.

    The file EXISTS and is the real one the leg was built from, which is the case that matters: a
    refusal that only fires on a missing path would let the plausible mistake through.
    """
    run, _ = _generate(tmp_path, plan_directory)
    named = {"-p": run / "leg" / "topology.pdb", "-s": run / "leg" / "system.xml",
             "-groupfile": run / "leg" / "leg.json"}[flag]
    assert _run_md_run(run, "--window", "w000", "--cpu", flag, str(named)) == 2
    message = capsys.readouterr().err
    assert flag in message and "alchemical ladder" in message and "leg/" in message
    assert not (run / "leg" / "windows").exists()


def test_a_ladder_refuses_x(tmp_path, plan_directory, capsys):
    """One name cannot describe N windows' streams, and a "prefix" option is how that pretence
    grows a meaning nobody tested."""
    run, _ = _generate(tmp_path, plan_directory)
    assert _run_md_run(run, "--window", "w000", "--cpu", "-x", "traj.nc") == 2
    assert "-x" in capsys.readouterr().err


@pytest.mark.parametrize("selector,fragment", [("w009", "does not exist"),
                                               ("9", "does not exist"),
                                               ("banana", "not a window")])
def test_a_window_that_does_not_exist_is_refused_not_skipped(tmp_path, plan_directory, capsys,
                                                             selector, fragment):
    """A campaign is complete when every window is, so silently running fewer would report a
    complete campaign that sampled part of the path."""
    run, _ = _generate(tmp_path, plan_directory)
    assert _run_md_run(run, "--window", selector, "--cpu", "--check") == 2
    assert fragment in capsys.readouterr().err


@pytest.mark.parametrize("selector,expected", [(None, ["w000", "w001", "w002", "w003"]),
                                               ("w002", ["w002"]),
                                               ("2", ["w002"]),
                                               ("0-2", ["w000", "w001", "w002"]),
                                               ("3,0", ["w003", "w000"])])
def test_the_window_selector_mirrors_ais_paths(tmp_path, plan_directory, selector, expected):
    from md_tools.alchemy.generated import select_windows

    run, _ = _generate(tmp_path, plan_directory)
    resolved = yaml.safe_load((run / "resolved.config").read_text())
    assert select_windows(resolved, selector) == expected


def test_a_script_and_a_declaration_that_disagree_are_refused(tmp_path, plan_directory):
    """The generated file declares what it runs. A directory holding a script for one ladder and
    a configuration describing another is regenerated, not run."""
    from md_tools.alchemy.generated import GeneratedWindowError, run_generated_window

    run, _ = _generate(tmp_path, plan_directory)
    text = (run / "resolved.config").read_text().replace("number_of_windows: 4",
                                                         "number_of_windows: 3")
    (run / "resolved.config").write_text(text, encoding="utf-8")
    with pytest.raises(GeneratedWindowError, match="different ladders"):
        run_generated_window(run / "w003.py", "w003", ["--cpu", "--check"])


# --- the parser change must not have loosened any other protocol ------------------------------

def test_every_other_protocol_still_requires_p_and_s(tmp_path, capsys):
    """`-p` and `-s` moved off argparse so a ladder could run without them. They are still
    required for everything else, and now refused BY NAME with the protocol speaking."""
    config = tmp_path / "cMD.in"
    config.write_text("&cntrl\n  protocol = cMD,\n  stage = min,\n/\n", encoding="utf-8")
    assert md_run_main(["-i", str(config), "-odir", str(tmp_path / "out")]) == 2
    assert "-p/--topology is required" in capsys.readouterr().err
