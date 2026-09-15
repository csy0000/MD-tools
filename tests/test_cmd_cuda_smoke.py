"""Both cMD bundle layouts, run on CUDA through the real commands.

`build-md` emits a cMD protocol in two shapes: a SPLIT chain of one script per stage, driven by
`run.sh`, and an ALL-IN-ONE `md.py` that carries the same stages in a single file. The acceptance
list names both, and until now only their GENERATION was tested. Generation proves the files are
written and compile; it does not prove the chain runs, that a later stage picks up the state the
previous one left, or that each stage records its own completion.

Shortened but meaningful: a tiny implicit-solvent peptide, a few hundred steps per stage. Implicit
so there is no box to solvate and no barostat to schedule -- the stage chain is what is under test,
not the solvent model. Real CUDA, because device selection and context creation are exactly the
paths a CPU run would not exercise.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from md_tools.build.record import read_record

REPO = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

#: Short enough to be a test, long enough that every stage actually integrates.
STAGES = {"minimization_iterations": 25, "restrained_nvt_steps": 50,
          "restrained_npt_steps": 50, "unrestrained_npt_steps": 50,
          "production_steps": 200}
REPORTING = {"crd_printout_solute": 20, "info_printout": 100, "checkpoint_printout": 200}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One implicit peptide system, built once and shared by both layouts."""
    work = tmp_path_factory.mktemp("cmd-cuda")
    (work / "build").mkdir(exist_ok=True)
    ala = REPO / "tests" / "data" / "ALA.pdb"
    if not ala.is_file():
        pytest.skip("no ALA fixture")
    # GBn2: no box, no barostat, no NPT stage -- the stage chain is what is under test.
    (work / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    build = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-top", "-i", str(ala),
         "-os", "build/built.xml", "-op", "build/built.pdb", "-log", "build/built.log",
         "--config", str(work / "sys.config")],
        cwd=work, capture_output=True, text=True, timeout=1800)
    assert build.returncode == 0, build.stdout + build.stderr
    return work


def _generate(work: Path, out: str, *extra: str) -> Path:
    config = work / f"{out}.config"
    config.write_text(yaml.safe_dump(
        {"protocol": "cMD", "solvent": "implicit",
         # `dynamics.platform` is retired; CUDA is the machine default and
         # `test_the_split_chain_ran_on_cuda` asserts the resolved Context.
         "dynamics": {"seed": 7},
         "stages": STAGES, "reporting": REPORTING}, sort_keys=False), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "md_tools.cli.md_openmm", "build-md", "-odir", f"./{out}-run1",
         "--config", str(config), *extra],
        cwd=work, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return work / f"{out}-run1"


def _run(directory: Path):
    return subprocess.run(["bash", "run.sh", "../build/built.pdb", "../build/built.xml"],
                          cwd=directory, capture_output=True, text=True, timeout=3600)


@pytest.fixture(scope="module")
def split(built):
    directory = _generate(built, "split")
    ran = _run(directory)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    return directory


@pytest.fixture(scope="module")
def all_in_one(built):
    directory = _generate(built, "single", "--all-in-one")
    ran = _run(directory)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    return directory


def test_the_split_chain_runs_every_stage_to_completion(split):
    """Each stage writes its own record, and each one says it finished.

    An implicit chain is minimisation, restrained NVT, a second restrained NVT and free NVT before
    production -- no NPT, because a system with no box has no volume to control.
    """
    logs = sorted(p.name for p in split.glob("*.log") if p.name != "build-md.log")
    assert logs, f"no stage records were written: {sorted(p.name for p in split.iterdir())}"
    for log in logs:
        record = read_record(split / log)
        assert record["status"] == "completed", f"{log}: {record.get('status')}"


def test_the_split_chain_ran_on_cuda(split):
    """A CPU fallback would still finish, and would prove nothing about the platform used."""
    for log in split.glob("*.log"):
        if log.name == "build-md.log":
            continue
        record = read_record(log)
        platform = (record.get("platform") or {})
        name = platform.get("name") if isinstance(platform, dict) else platform
        assert name == "CUDA", f"{log.name} ran on {name!r}"


def test_production_continued_from_the_state_equilibration_left(split):
    """The chain is only a chain if each stage consumes the previous stage's state.

    This is the property `run.sh` exists for. A stage that silently restarted from the built
    system would produce a plausible trajectory from the wrong configuration.
    """
    production = read_record(split / "cMD.log")
    parent = (production.get("inputs") or {}).get("continued_from")
    assert parent, "production records no parent state"
    assert parent.get("sha256"), "the parent is named but not pinned by content"


def test_the_all_in_one_layout_runs_the_same_stages_in_one_file(all_in_one):
    """One file, same chain. The layout is a packaging choice, not a different protocol."""
    assert (all_in_one / "md.py").is_file()
    assert not list(all_in_one.glob("eq_*.py")), "an all-in-one bundle must not emit stage scripts"
    records = [read_record(p) for p in all_in_one.glob("*.log") if p.name != "build-md.log"]
    assert records, "the all-in-one run wrote no record"
    assert all(r["status"] == "completed" for r in records), [r.get("status") for r in records]


def test_both_layouts_produce_a_trajectory(split, all_in_one):
    """AMBER NetCDF, not DCD -- and the SOLUTE stream, which every stage writes.

    The glob was `*.dcd`, the only format a stage could honestly produce while it used OpenMM's
    own reporter. It now writes AMBER NetCDF through MD-tools' appending writer, which is the
    only format that carries an atom subset -- and a solute-only stream is exactly that.
    """
    for directory in (split, all_in_one):
        produced = list(directory.glob("solute_*.nc"))
        assert produced, f"{directory.name} wrote no trajectory"
        assert all(p.stat().st_size > 0 for p in produced), produced
