"""A generated script is an entry point, not a copy of the implementation.

The failure this shape prevents: when the behaviour lived in the generated files, fixing a defect
meant regenerating every project ever produced, and two directories generated a month apart ran
different code while claiming the same protocol. Now there is one installed implementation, and
the generated file says only which workflow and which stage.

`resolved.config` beside the script is the single resolved declaration. It is strictly validated
at execution and bound into the checkpoint fingerprint, so a hand edit is refused rather than
half-applied.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

CONFIGS = {
    "cMD": {"protocol": "cMD", "solvent": "explicit", "stages": {"production_steps": 10}},
    "REST2": {"protocol": "REST2", "solvent": "explicit",
              "rest2": {"number_of_replicas": 2, "number_of_exchanges": 2}},
    "rREST2": {"protocol": "rREST2", "solvent": "explicit",
               "rest2": {"number_of_replicas": 2, "number_of_exchanges": 2},
               "reservoir": {"enabled": True, "path": "./reservoir"}},
    "AIS": {"protocol": "AIS", "solvent": "explicit",
            "ais": {"number_of_paths": 2, "switching_steps": 20,
                    "observation_interval_steps": 10},
            "ais_source": {"trajectory": "../source.dcd"}},
}


def _generate(tmp_path: Path, protocol: str, *extra: str, into: str | None = None) -> Path:
    config = tmp_path / f"{protocol}.config"
    config.write_text(yaml.safe_dump(CONFIGS[protocol], sort_keys=False), encoding="utf-8")
    out = tmp_path / (into or protocol)
    done = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-md", "-odir", str(out),
         "--config", str(config), *extra],
        capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return out


def _scripts(directory: Path):
    return [p for p in sorted(directory.glob("*.py"))]


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    work = tmp_path_factory.mktemp("compact")
    directories = {name: _generate(work, name) for name in CONFIGS}
    directories["all_in_one"] = _generate(work, "cMD", "--all-in-one",
                                          into="cMD_all_in_one")
    return directories


# --- the acceptance criteria --------------------------------------------------------------------

def test_no_generated_script_defines_a_function_or_a_class(generated):
    """A definition in a generated file is implementation, and implementation belongs in the
    package where it has one version."""
    for name, directory in generated.items():
        for script in _scripts(directory):
            tree = ast.parse(script.read_text(encoding="utf-8"))
            defined = [n.name for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            assert not defined, f"{name}/{script.name} defines {defined}"


def test_no_generated_script_parses_arguments_or_imports_openmm(generated):
    """Argument parsing and OpenMM are the runtime's job. A script that imported OpenMM would be
    doing the work rather than asking for it."""
    for name, directory in generated.items():
        for script in _scripts(directory):
            text = script.read_text(encoding="utf-8")
            body = text.split('"""', 2)[-1]                # ignore the docstring
            assert "argparse" not in body, f"{name}/{script.name} parses arguments"
            assert "import openmm" not in body and "from openmm" not in body, \
                f"{name}/{script.name} imports OpenMM"


def test_no_generated_script_constructs_the_physics(generated):
    """No reporter, restraint, barostat, scaler or exchange rule is built in a generated file."""
    forbidden = ("CustomExternalForce", "MonteCarloBarostat", "LangevinMiddleIntegrator",
                 "Simulation(", "DCDReporter", "StateDataReporter", "TauSwitcher",
                 "build_scaled_system", "NeighbouringExchangeRule", "ReservoirRefreshRule")
    for name, directory in generated.items():
        for script in _scripts(directory):
            body = script.read_text(encoding="utf-8").split('"""', 2)[-1]
            for token in forbidden:
                assert token not in body, f"{name}/{script.name} constructs {token}"


def test_each_generated_script_is_one_import_and_one_call(generated):
    """The whole body: import the stable API, call it. Anything else is drift."""
    for name, directory in generated.items():
        for script in _scripts(directory):
            tree = ast.parse(script.read_text(encoding="utf-8"))
            statements = [n for n in tree.body if not (isinstance(n, ast.Expr)
                                                       and isinstance(n.value, ast.Constant))]
            kinds = [type(n).__name__ for n in statements]
            assert kinds == ["ImportFrom", "Raise"], f"{name}/{script.name}: {kinds}"
            imported = statements[0]
            assert imported.module.startswith("md_tools."), imported.module


def test_no_generated_script_embeds_a_second_declaration(generated):
    """`resolved.config` is the single declaration. A `STAGE = {...}` or `LADDER = {...}` literal
    in the script is a second one that can disagree with it after a hand edit."""
    for name, directory in generated.items():
        for script in _scripts(directory):
            body = script.read_text(encoding="utf-8").split('"""', 2)[-1]
            for token in ("STAGE = {", "LADDER = {", "STAGES = ", "RUN = {"):
                assert token not in body, f"{name}/{script.name} embeds {token}"


def test_every_generated_directory_carries_exactly_one_resolved_config(generated):
    for name, directory in generated.items():
        assert (directory / "resolved.config").is_file(), f"{name} has no resolved.config"


def test_no_generated_executable_names_a_checkout_or_machine_path(generated):
    """A generated directory must be movable: nothing it EXECUTES may name where it was made.

    Scoped to the scripts, `run.sh` and `resolved.config` -- the files that are read to run. Not
    `build-md.log`, which is the machine record and legitimately records the absolute path of the
    input it was given; that is provenance, and stripping it would make the record less useful
    without making the directory more portable.
    """
    for name, directory in generated.items():
        for path in sorted(directory.iterdir()):
            if path.name == "build-md.log" or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for leak in (str(REPO), "/home/", "/data3/", "site-packages"):
                assert leak not in text, f"{name}/{path.name} names {leak}"


def test_every_generated_script_compiles(generated):
    import py_compile

    for directory in generated.values():
        for script in _scripts(directory):
            py_compile.compile(str(script), doraise=True)


def test_ais_carries_no_equilibration_chain(generated):
    """AIS consumes an ensemble that already exists; generating one would start it in the wrong
    distribution."""
    names = {p.name for p in _scripts(generated["AIS"])}
    assert names == {"AIS.py"}, names


def test_the_all_in_one_bundle_is_one_script(generated):
    names = {p.name for p in _scripts(generated["all_in_one"])}
    assert names == {"md.py"}, names


def test_a_moved_directory_still_runs(generated, tmp_path):
    """`__file__`, not the working directory: the helpers find `resolved.config` beside the script.

    Run with `--check`, so this asserts the location and validation without integrating.
    """
    import shutil

    moved = tmp_path / "somewhere" / "else"
    moved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(generated["cMD"], moved)
    done = subprocess.run([sys.executable, str(moved / "min.py"), "--help"],
                          capture_output=True, text=True, timeout=300, cwd=str(tmp_path))
    # `--help` reaches argparse inside the runtime, which means resolved.config was found and
    # resolved from the script's own location rather than from the working directory.
    assert done.returncode == 0, done.stdout + done.stderr
    # SUPERSEDED: this looked for `--platform`, which is retired -- the platform is
    # machine.openmm.platform in the user configuration now. `--cpu` is the per-run override and
    # serves the same purpose here: proving argparse inside the runtime was reached.
    assert "-p" in done.stdout and "--cpu" in done.stdout


def test_editing_resolved_config_is_refused_by_the_strict_resolver(generated, tmp_path):
    """It is validated at EXECUTION, not only at generation."""
    import shutil

    work = tmp_path / "edited"
    shutil.copytree(generated["cMD"], work)
    config = work / "resolved.config"
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["stages"]["producton_steps"] = 10            # deliberate typo
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    done = subprocess.run([sys.executable, "min.py", "--help"],
                          cwd=work, capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "producton_steps" in (done.stdout + done.stderr)
