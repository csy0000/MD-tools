"""Every command in the documentation, executed with the flag meanings it declares.

A help test that checks the flag NAMES appear proves nothing about what they mean. `-x` appeared in
the help of the version where it named the serialised System and of the version where it names the
trajectory, and only one of those is right. So these run the commands.

The commands are extracted from the documentation itself rather than copied here, so a page that
drifts fails this rather than quietly documenting something that no longer works.

Marked slow, not gpu: they run with `--cpu` where they integrate at all, because what is under
test is the command surface. The real CUDA and MPI evidence is `test_md_run_mpi_gpu.py`.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

#: ONE worker for this module. Its fixtures build a wheel, a venv and a system; scattered across
#: workers they are built once per worker that draws a test.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("documented-commands")]

#: The pages whose command blocks are contractual.
PAGES = [REPO / "README.md", REPO / "CLAUDE.md", REPO / "docs" / "md-run.md",
         REPO / "docs" / "openmm_methods" / "cMD" / "README.md",
         REPO / "docs" / "openmm_methods" / "REST2" / "README.md",
         REPO / "docs" / "openmm_methods" / "AIS" / "README.md"]


def _md_run_commands(text: str) -> list[list[str]]:
    """Every `md-openmm md-run` INVOCATION on a page, with line continuations joined.

    Only lines inside a shell fence, and only ones that carry `-i`. Prose mentions the command by
    name -- "run.sh drives `md-openmm md-run`", "md-openmm md-run ~ pmemd.cuda", the summary table
    of subcommands -- and those are not commands to execute.
    """
    found = []
    for block in re.findall(r"```(?:bash|sh)\n(.*?)```", text, flags=re.S):
        for line in block.replace("\\\n", " ").splitlines():
            stripped = line.strip()
            if "md-openmm md-run" not in stripped or stripped.startswith("#"):
                continue
            words = shlex.split(stripped.split("md-openmm md-run", 1)[1])
            if "-i" in words:
                found.append(words)
    return found


def test_the_documentation_uses_the_amber_flag_meanings_everywhere():
    """`-s` is always the System and `-x` is always the trajectory. Checked against the pages.

    This is the assertion that would have caught the whole `-x = built.xml` mistake at the moment
    it entered the documentation: every documented `-s` value looks like a System, and every
    documented `-x` value looks like a trajectory.

    EXACTLY ONE OF `-s` AND `--groupfile`, and that is checked here rather than assumed. This test
    used to read `parsed.system.endswith(".xml")` for every command, which encoded "every
    documented invocation carries `-s`" -- true only while a ladder had no group file. A ladder's
    rungs are each a pre-scaled System named on its own line of the group file, so there is no
    single `-s` for the launch to carry and one would claim one Hamiltonian for every rung; the
    runtime refuses whichever of the two was not asked for. The old form did not merely miss that
    case, it CRASHED on it (`AttributeError` on `None`), and because this module is marked `slow`
    no fast lane ran it -- so a correct, generated-by-`run.sh` ladder command sat in the README
    failing this test unnoticed. Checking the exclusion also catches a command carrying BOTH,
    which the previous assertion never could.
    """
    from md_tools.run.main import md_run_parser

    seen = 0
    for page in PAGES:
        for argv in _md_run_commands(page.read_text(encoding="utf-8")):
            seen += 1
            parsed = md_run_parser().parse_args(argv)      # SystemExit here = a broken example
            if parsed.groupfile:
                assert parsed.system is None, (
                    f"{page.name}: a ladder names its rungs in the group file, so it must not "
                    f"also pass -s: {argv}")
            else:
                assert parsed.system is not None and parsed.system.endswith(".xml"), (
                    page.name, argv)
            if parsed.trajectory:
                assert Path(parsed.trajectory).suffix in (".dcd", ".nc"), (page.name, argv)
            if parsed.output and parsed.log:
                assert parsed.output != parsed.log, (page.name, argv)
    assert seen >= 8, f"only {seen} documented md-run commands were found; the extractor is wrong"


def test_no_page_documents_a_retired_flag():
    """`--platform` and `-x built.xml` are retired. A page that still shows them is stale text."""
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        for retired in ("--platform ", "-x built.xml", "-x ../built.xml", "--trajectory "):
            assert retired not in text, f"{page.name} still documents {retired!r}"


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    """A wheel, installed into a clean environment, exercised from outside the checkout."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    from tests.wheel_build import build_wheel_from_copy

    root = tmp_path_factory.mktemp("wheel")
    # From a copy, and a failure is a failure. This used to build in the checkout and SKIP on any
    # error, so two xdist workers colliding in `<repo>/build/` reported "the wheel could not be
    # built here" -- a concurrency bug presented as a fact about the machine.
    wheel = build_wheel_from_copy(REPO, root)

    environment = root / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
                   check=True, capture_output=True, timeout=600)
    # `--force-reinstall` because the development environment installs this package editable, and
    # a `--system-site-packages` venv would otherwise see it as already satisfied and install
    # nothing -- leaving a test that believes it is exercising a wheel and is exercising the
    # checkout. With it, the venv's own site-packages wins and `md_tools.__file__` proves it.
    subprocess.run([str(environment / "bin" / "pip"), "install", "--no-deps",
                    "--force-reinstall", str(wheel)],
                   check=True, capture_output=True, timeout=600)

    work = root / "outside"
    work.mkdir()
    return environment / "bin", work


def _run(installed, *args, cwd=None, timeout=1800):
    """The INSTALLED command, with `PYTHONPATH` scrubbed so it cannot reach the checkout."""
    import os

    binaries, work = installed
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    return subprocess.run([str(binaries / "md-openmm"), *args], cwd=cwd or work,
                          capture_output=True, text=True, timeout=timeout, env=environment)


def test_the_wheel_installs_one_executable_and_four_subcommands(installed):
    binaries, _ = installed
    scripts = sorted(p.name for p in binaries.iterdir()
                     if p.name.startswith("md") or "openmm" in p.name)
    assert scripts == ["md-openmm"], scripts

    done = _run(installed, "-h")
    assert done.returncode == 0, done.stderr
    for name in ("build-top", "build-md", "md-run", "data-register"):
        assert name in done.stdout, name


def test_md_run_resolves_outside_the_checkout(installed):
    """`md_tools` must come from site-packages, not from a source tree that happens to be near.

    `PYTHONPATH` is scrubbed deliberately. `tests/conftest.py` puts `src` on it so that
    subprocesses in the rest of the suite exercise the working tree, which is the right default
    everywhere except here: this is the one test whose subject is the INSTALLED artefact, and
    inheriting that variable would have it quietly import the checkout and pass.
    """
    import os

    binaries, work = installed
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    done = subprocess.run([str(binaries / "python"), "-c",
                           "import md_tools; print(md_tools.__file__)"],
                          cwd=work, capture_output=True, text=True, timeout=300, env=environment)
    assert done.returncode == 0, done.stderr
    assert "site-packages" in done.stdout, done.stdout
    assert str(REPO / "src") not in done.stdout, done.stdout


@pytest.fixture(scope="module")
def generated(installed):
    """A built system and a generated project, made ONCE with the installed commands.

    Every test below that reads `md_script` takes this fixture. They used to read the directory
    `test_the_canonical_stage_command_runs_against_the_installed_wheel` left behind, so which of
    them passed depended on whether xdist had put that test on the same worker first -- and a
    refusal test failed with `FileNotFoundError: .../outside/md_script`, a statement about test
    ordering reported as one about the command.
    """
    _binaries, work = installed
    (work / "ALA.pdb").write_bytes(ALA.read_bytes())
    (work / "build").mkdir(exist_ok=True)

    built = _run(installed, "build-top", "-i", "ALA.pdb", "-os", "build/built.xml",
                 "-op", "build/built.pdb", "-log", "build/built.log")
    assert built.returncode == 0, built.stdout[-2000:] + built.stderr[-2000:]

    (work / "c.config").write_text(yaml.safe_dump({
        "protocol": "cMD", "solvent": "explicit",
        "stages": {"minimization_iterations": 25, "restrained_nvt_steps": 20,
                   "restrained_npt_steps": 20, "unrestrained_npt_steps": 20,
                   "production_steps": 40},
        "reporting": {"crd_printout_solute": 20, "info_printout": 20,
                      "checkpoint_printout": 40}}, sort_keys=False), encoding="utf-8")
    made = _run(installed, "build-md", "-odir", "cMD-run1", "--config", "c.config")
    assert made.returncode == 0, made.stdout[-2000:] + made.stderr[-2000:]
    return work / "cMD-run1"


def test_the_canonical_stage_command_runs_against_the_installed_wheel(installed, generated):
    """The exact command `docs/md-run.md` opens with, on the CPU, end to end."""
    _binaries, work = installed
    script = generated
    done = _run(installed, "md-run", "-i", "../input/cMD.in", "-p", "../build/built.pdb", "-s", "../build/built.xml",
                "-o", "cMD.out", "-x", "cMD.dcd", "-r", "cMD.xml", "-log", "cMD.log", "--cpu",
                cwd=script)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    # Every flag produced the artefact its documentation says it names.
    from md_tools.build.record import read_record
    from md_tools.openmm.trajectory import detect_trajectory_format

    # `-x cMD.dcd` was given explicitly, so the stage writes THAT name -- and in DCD, because
    # the writer follows the suffix. A stage left to name its own streams writes
    # `solute_prod<N>.nc` in NetCDF; both are covered, and neither puts one format in the other's
    # name.
    assert detect_trajectory_format(script / "cMD.dcd") == "dcd"
    assert (script / "cMD.xml").is_file()
    assert "status               completed" in (script / "cMD.out").read_text(encoding="utf-8")
    assert read_record(script / "cMD.log")["status"] == "completed"
    # -s was not overwritten by -x, which is the accident the old mapping invited.
    assert (work / "build" / "built.xml").read_text(encoding="utf-8").lstrip().startswith("<")


def test_the_generated_project_names_no_checkout_path(generated):
    """A generated directory has to be movable, and must not import a source tree."""
    script = generated
    for path in list(script.glob("*.py")) + list(script.glob("*.in")) + \
            [script / "run.sh", script / "resolved.config"]:
        text = path.read_text(encoding="utf-8")
        assert str(REPO) not in text, path.name
        assert "site-packages" not in text, path.name


def test_the_installed_command_refuses_the_retired_platform_flag(installed, generated):
    done = _run(installed, "md-run", "-i", "../input/cMD.in", "-p", "../build/built.pdb", "-s", "../build/built.xml",
                "--platform", "CUDA", cwd=generated)
    assert done.returncode != 0
    assert "platform" in done.stderr.lower(), done.stderr


def test_the_installed_command_refuses_a_system_in_the_trajectory_flag(installed, generated):
    done = _run(installed, "md-run", "-i", "../input/cMD.in", "-p", "../build/built.pdb", "-s", "../build/built.xml",
                "-x", "../build/built.xml", cwd=generated)
    assert done.returncode != 0
    assert "-x" in done.stderr and "-s" in done.stderr, done.stderr


def test_the_generated_run_sh_examples_match_the_flag_contract(generated):
    """run.sh is a documented command too, and it is the one most people actually execute."""
    text = (generated / "run.sh").read_text(encoding="utf-8")
    assert '-s "${SYSTEM}"' in text and '-x "${SYSTEM}"' not in text, text
    # NO `-x`: a stage writes two coordinate streams and names them itself. See
    # `test_md_run_inputs.py` for the same contract stated there.
    # No `-x` naming a STAGE trajectory: a stage writes two coordinate streams and
    # names them itself. The ladder line legitimately keeps `-x REST2.nc`, which is
    # the exchange record rather than a trajectory, so the check is specific.
    assert not re.search(r"-x \S+\.dcd", text), text
    # `-r`, `-chk`, `-o` and `-log` ARE OPTIONAL, and run.sh leaves them off deliberately.
    #
    # Two files for two readers is still the contract -- `.out` is what a person tails, `.log` is
    # the machine record -- but md-run names both itself, inside the directory `-odir` points at
    # (`main.py`: `args.output or str(out_dir / f"{name}.out")`). So a stage writes
    # `../min/min.out` and `../min/min.log` without run.sh restating either.
    #
    # Spelling them here would be worse than redundant: a value that IS given is taken verbatim
    # against the WORKING directory, not against `-odir`, so `-o min.out` from a run root would
    # put the minimisation's output beside run.sh instead of in `min/`. That is the defect the
    # per-stage `-odir` removes, so the assertion is inverted rather than dropped.
    assert re.search(r"-odir \S+", text), text
    assert not re.search(r"-o \w+\.out", text), (
        "run.sh names an output path explicitly; md-run resolves a given -o against the working "
        "directory, so the file would land beside run.sh rather than in -odir")
