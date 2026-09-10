"""The rREST2 reservoir is validated BEFORE the run directory exists.

WHAT WAS WRONG

    `reservoir_declaration_text` was called at helper-publication time, which is after
    `out.mkdir()`. So a reservoir that did not exist, held no frames, or could not be read as
    phase space was discovered with `-odir` already created -- and an empty directory that says a
    run was started here is exactly what the next `--resume` finds.

    The same source was then opened a SECOND time by the driver, after `_begin` had created the
    analysis file and the per-state trajectories, to rediscover facts the first reading had
    already established. Two readings of one file, either of which could be the one that refuses,
    and only one of them happening early enough to matter.

    Both are now one reading, in the preflight, whose result is carried on `LadderPreflight` and
    consumed by the publication step.

PLATFORM_POLICY_EXEMPTION: the Reference platform here builds one serialized State to hand to
`-c`; it propagates nothing. The ladder itself runs through the generated script under `--cpu`,
because what is under test is which checks run BEFORE output exists -- a question about ordering,
not about arithmetic, and identical on every platform.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A generated rREST2 tree whose reservoir path this module varies per test."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("reservoir-preflight")
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    (root / "rREST2.config").write_text(yaml.safe_dump({
        "protocol": "rREST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 5,
                   "restrained_npt_steps": 5, "unrestrained_npt_steps": 5,
                   "production_steps": 5},
        "rest2": {"number_of_replicas": 2, "exchange_interval_steps": 5,
                  "number_of_exchanges": 2},
        "reservoir": {"enabled": True, "path": "../reservoir.nc"},
        "reporting": {"crd_printout_solute": 5, "info_printout": 5, "checkpoint_printout": 5},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./rREST2", "--config", str(root / "rREST2.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    return root


def _top_rung(project: Path, tau_max: float = 0.5):
    """The scaled System the ladder's top rung IS, and the selection it was scaled with.

    Built through the same helpers the ladder uses, because the reservoir's Hamiltonian identity
    has to match that System exactly -- which is the whole point of the check. Restating the
    identity by hand would test that the fixture and the checker agree about a dictionary, not
    that the source samples the distribution it will refresh.
    """
    from md_tools.remd.generated import solute_document
    from md_tools.run.preflight import check_scaling_plan, load_inputs

    loaded = load_inputs(str(project / "built.pdb"), str(project / "built.xml"))
    document = solute_document(loaded.pdb.topology, loaded.system, route=None)
    span = document.get("solute_atom_range")
    if span and document.get("solute_atom_indices_are_contiguous", False):
        indices = list(range(int(span[0]), int(span[1]) + 1))
    else:
        indices = list(range(int(document["n_solute_atoms"])))
    excluded = [tuple(int(a) for a in pair)
                for pair in (document.get("rest2") or {}).get("omega_excluded_bonds", [])]
    _audit, scaled = check_scaling_plan(loaded, solute_indices=indices, excluded_bonds=excluded,
                                        tau=tau_max, where="fixture")
    return scaled, indices, excluded


def _write_reservoir(path: Path, project: Path, *, frames: int = 4, tau_max: float = 0.5,
                     temperature_k: float = 300.0):
    """A genuine phase-space file carrying the top rung's Hamiltonian identity.

    What a real fixed-tau cMD at the ladder's `tau_max` would have written. A file without this
    identity is correctly refused -- the preflight cannot show it samples the distribution it
    would refresh -- so a fixture that omitted it would only ever test the refusal.
    """
    import mdtraj

    from md_tools.md.phase_space import PhaseSpaceWriter
    from md_tools.rest2 import identity as hamiltonian_identity

    scaled, indices, excluded = _top_rung(project, tau_max)
    positions = mdtraj.load(str(project / "built.pdb")).xyz[0]
    identity = {"hamiltonian": hamiltonian_identity.identity_record(
        scaled, tau=tau_max, temperature_k=temperature_k, ensemble="NVT",
        solute_indices=indices, excluded_bonds=excluded)}

    writer = PhaseSpaceWriter(path, n_atoms=positions.shape[0], periodic=False, identity=identity)
    try:
        for index in range(frames):
            writer.append(positions=positions, velocities=positions * 0.0, box=None,
                          step=index * 5, time_ps=float(index))
    finally:
        writer.close()


@pytest.fixture(scope="module")
def initial_state(project):
    """A real serialized State (positions, velocities, box), for `-c`.

    A ladder never reaches propagation without one, and only the last test here runs that far.
    """
    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(project / "built.pdb"))
    system = XmlSerializer.deserialize((project / "built.xml").read_text(encoding="utf-8"))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    state = context.getState(getPositions=True, getVelocities=True,
                             enforcePeriodicBox=system.usesPeriodicBoundaryConditions())
    path = project / "initial_state.xml"
    path.write_text(XmlSerializer.serialize(state), encoding="utf-8")
    return path


def _run(project: Path, destination: Path, *extra):
    import os

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"),
         *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    environment["MD_TOOLS_CONFIG"] = str(user)
    return subprocess.run(
        [sys.executable, str(project / "rREST2" / "rREST2.py"),
         "-p", str(project / "built.pdb"), "-s", str(project / "built.xml"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "rREST2", capture_output=True, text=True, timeout=900, env=environment)


def test_a_missing_reservoir_is_refused_before_the_output_directory_exists(project, tmp_path):
    """THE guarantee: nothing is created by a run that cannot start.

    The refusal itself is not the interesting part -- the baseline refused too, just later. What
    this pins is that `-odir` does not exist afterwards, so the next invocation finds no evidence
    that a run was ever begun here.
    """
    reservoir = project / "reservoir.nc"
    if reservoir.exists():
        reservoir.unlink()
    destination = tmp_path / "no-reservoir"

    done = _run(project, destination)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "reservoir" in (done.stdout + done.stderr).lower()
    assert not destination.exists(), (
        f"{destination} was created by a run that could not start; the next --resume would find "
        f"it and believe a ladder had been started here")


def test_an_empty_reservoir_is_refused_before_the_output_directory_exists(project, tmp_path):
    """A file that exists and holds nothing: `exists()` is not `is usable`."""
    import netCDF4

    from md_tools.md.phase_space import PhaseSpaceWriter

    reservoir = project / "reservoir.nc"
    writer = PhaseSpaceWriter(reservoir, n_atoms=22, periodic=False, identity={"tau": 0.5})
    writer.close()
    # CLOSED, not merely opened. A netCDF handle left open holds the file, and the next test in
    # this module rewrites this very path -- which failed with a PermissionError only when the
    # whole lane ran in one process, never when this file ran alone.
    dataset = netCDF4.Dataset(str(reservoir))
    try:
        assert dataset.dimensions["frame"].size == 0
    finally:
        dataset.close()

    destination = tmp_path / "empty-reservoir"
    done = _run(project, destination)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "no frames" in (done.stdout + done.stderr).lower(), done.stdout + done.stderr
    assert not destination.exists()


def test_check_reads_the_reservoir_and_still_creates_nothing(project, tmp_path):
    """`--check` is the preflight and nothing else, so it must now see the reservoir too.

    Both halves matter: it has to actually READ the file (the baseline's --check returned before
    the declaration was ever built, so a missing reservoir passed --check and failed the run), and
    it must still write nothing at all.
    """
    _write_reservoir(project / "reservoir.nc", project)
    destination = tmp_path / "checked"

    done = _run(project, destination, "--check")
    assert done.returncode == 0, done.stdout + done.stderr
    assert not destination.exists(), "--check created the directory it was asked to check"


def test_the_preflight_carries_the_declaration_the_publication_step_writes(
        project, initial_state, tmp_path):
    """One reading of the source, not two: the text is built once and then consumed.

    A real run, then the published `reservoir.yaml` compared against what a fresh preflight
    produces for the same inputs -- they must be byte-identical, which is what makes "the driver
    consumes the prepared declaration" a checkable claim rather than a comment.
    """
    _write_reservoir(project / "reservoir.nc", project)
    destination = tmp_path / "run"

    done = _run(project, destination, "-c", str(initial_state))
    assert done.returncode == 0, done.stdout + done.stderr
    written = (destination / "reservoir.yaml").read_text(encoding="utf-8")
    # Rank 0 publishes helpers with a content-address header, which is how a stale helper from an
    # incompatible earlier run is refused rather than reused. The declaration is what follows it.
    header, _, published = written.partition("\n")
    assert header.startswith("# md-tools-helper-sha256:"), written[:200]

    from md_tools.remd.generated import reservoir_declaration_text

    ladder = {"reservoir": {"enabled": True, "path": str(project / "reservoir.nc"),
                            "velocities": None, "refresh_interval_exchanges": 1},
              "dynamics": {"seed": yaml.safe_load(published)["random_seed"]}}
    assert reservoir_declaration_text(ladder, destination) == published, (
        "the published declaration is not the one the preflight built")

    # And the facts in it are the ones the file actually holds, not restated configuration.
    source = yaml.safe_load(published)["source"]
    assert source["frames"] == 4
    assert source["start_time_ps"] == 0.0 and source["end_time_ps"] == 3.0
