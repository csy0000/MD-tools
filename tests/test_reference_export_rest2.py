"""A REST2 bundle must run without md_tools, and must reproduce the ladder exchange for exchange.

PLATFORM_POLICY_EXEMPTION: a four-rung ALA ladder, ten exchanges of fifty steps, on the CPU. What
is under test is whether the exported bundle makes the SAME decisions as the engine, which is
platform-independent.

WHY THE DECISIONS AND NOT THE ACCEPTANCE RATE

    A runner that swaps the wrong pairs still produces a plausible acceptance rate. The quantity
    that cannot be accidentally right is the state-to-walker mapping after every exchange:
    matching it requires the per-rung integrator seeds (`seed + 977*rung`), all four reduced
    potentials, the odd/even sweep phase, the acceptance criterion and the RNG substream to be
    identical. Any one of them wrong and the mapping diverges within two or three exchanges.

    That is the check the cMD export did not have, and it is why the cMD export shipped carrying
    the BUILD System instead of the integrated one -- its first test compared the bundle against
    a Context built in the test file, which shared all of the bundle's assumptions.

WHY BYTE-IDENTITY IS ASSERTED AS WELL

    The bundle does not reimplement the exchange; it carries md_tools' own modules, copied. So
    the strongest available statement is not "these agree" but "these are the same bytes", and
    that is what makes drift impossible rather than merely detectable.
"""
from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

ONE_THREAD = {**os.environ, "OPENMM_CPU_THREADS": "1"}
RUNGS, EXCHANGES, INTERVAL = 4, 10, 50

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    """A real four-rung ladder through the engine, and the bundle exported from it."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    if subprocess.run(["which", "mpirun"], capture_output=True).returncode != 0:
        pytest.skip("no mpirun on PATH; a ladder needs one rank per rung")

    root = tmp_path_factory.mktemp("rest2reference")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    assert subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800).returncode == 0

    (root / "REST2.config").write_text(
        "protocol: REST2\nsolvent: implicit\n"
        "dynamics: {timestep_fs: 2.0, temperature_K: 300.0, seed: 7}\n"
        "stages: {minimization_iterations: 0, restrained_nvt_steps: 100, restrained_npt_steps: 0,"
        " unrestrained_npt_steps: 0, production_steps: 500}\n"
        "reporting: {crd_printout_solute: 50, info_printout: 50, checkpoint_printout: 500}\n"
        f"rest2: {{number_of_replicas: {RUNGS}, tau_max: 0.5, "
        f"exchange_interval_steps: {INTERVAL}, number_of_exchanges: {EXCHANGES}, "
        "state_trajectory: true, rem_log: true, neighbour_acceptance_report: true}\n",
        encoding="utf-8")
    assert subprocess.run(CLI + ["build-md", "-odir", "./run", "--config",
                                 str(root / "REST2.config")],
                          cwd=root, capture_output=True, text=True, timeout=600).returncode == 0

    run = root / "run"
    equilibration = subprocess.run(
        CLI + ["md-run", "-i", "eq_nvt_posres.in", "-p", "../built.pdb", "-s", "../built.xml",
               "-r", "eq.xml", "-chk", "eq.chk", "-o", "eq.out", "-log", "eq.log",
               "-odir", ".", "--cpu"],
        cwd=run, capture_output=True, text=True, timeout=1800, env=ONE_THREAD)
    assert equilibration.returncode == 0, equilibration.stdout[-3000:] + equilibration.stderr[-3000:]

    done = subprocess.run(
        ["mpirun", "-n", str(RUNGS), *CLI, "md-run", "-ng", str(RUNGS), "-i", "REST2.in",
         "-p", "../built.pdb", "-s", "../built.xml", "-c", "eq.xml", "-x", "REST2.nc",
         "-r", "restart.json", "-o", "REST2.out", "-log", "REST2.log", "--cpu"],
        cwd=run, capture_output=True, text=True, timeout=3600, env=ONE_THREAD)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    from md_tools.reference import export_rest2_reference

    bundle = root / "bundle"
    manifest = export_rest2_reference(run, bundle, stage="REST2")
    return run, bundle, manifest


def test_the_vendored_modules_are_byte_identical_to_the_packages_own(ladder):
    """Not "equivalent": the same bytes. Drift is then impossible rather than detectable."""
    _run, bundle, manifest = ladder
    source = Path(__import__("md_tools.remd", fromlist=["__file__"]).__file__).parent
    for name in manifest["vendored"]:
        mine = hashlib.sha256((bundle / "ladder" / name).read_bytes()).hexdigest()
        theirs = hashlib.sha256((source / name).read_bytes()).hexdigest()
        assert mine == theirs, f"ladder/{name} is not md_tools/remd/{name}"


def test_the_bundle_never_imports_md_tools(ladder):
    """Enforced by blocking the import, not by reading the source."""
    _run, bundle, _manifest = ladder
    blocker = bundle / "_blocked.py"
    blocker.write_text(
        "import sys, runpy\n"
        "class B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'md_tools':\n"
        "            raise ImportError('bundle imported md_tools: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        f"sys.argv = ['run.py', '--exchanges', '{EXCHANGES}', '--platform', 'CPU']\n"
        "runpy.run_path('run.py', run_name='__main__')\n", encoding="utf-8")
    done = subprocess.run([sys.executable, str(blocker)], cwd=bundle, capture_output=True,
                          text=True, timeout=3600, env={**ONE_THREAD, "PYTHONPATH": ""})
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert "imported md_tools" not in done.stderr


def test_the_bundle_reproduces_the_engines_ladder_exchange_for_exchange(ladder):
    """The assertion this exporter exists for. See the module docstring for why it is the mapping."""
    run, bundle, _manifest = ladder
    assert (bundle / "exchange.csv").is_file(), (
        "run the bundle first; test_the_bundle_never_imports_md_tools does that")

    engine = {}
    with (run / "exchange.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            engine.setdefault(int(row["exchange"]), {})[int(row["state"])] = int(row["walker"])
    engine_mapping = {k: [v[s] for s in sorted(v)] for k, v in engine.items()}

    mapping = list(range(RUNGS))
    mine = {}
    with (bundle / "exchange.csv").open(encoding="utf-8") as handle:
        proposals = {}
        for row in csv.DictReader(handle):
            proposals.setdefault(int(row["exchange"]), []).append(
                (int(row["state_i"]), int(row["state_j"]), int(row["accepted"])))
    for attempt in sorted(proposals):
        for state_i, state_j, accepted in proposals[attempt]:
            if accepted:
                mapping[state_i], mapping[state_j] = mapping[state_j], mapping[state_i]
        mine[attempt] = list(mapping)

    assert engine_mapping, "the engine wrote no exchange record"
    differing = [k for k in sorted(engine_mapping) if engine_mapping[k] != mine.get(k)]
    assert not differing, (
        "the bundle and the engine disagree about which walkers occupy which states after "
        f"exchange(s) {differing}:\n"
        + "\n".join(f"  {k}: engine {engine_mapping[k]}  bundle {mine.get(k)}"
                    for k in differing[:5]))
