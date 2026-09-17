"""An rREST2 CV row holds the configuration the state PROPAGATED, never the reservoir sample.

THE SCIENTIFIC REGRESSION

    The ladder CV convention is pre-exchange: a row at an exchange boundary describes what the
    state actually propagated up to that step, before a swap or a reservoir replacement.

    The driver snapshotted only `state_to_walker`, which is enough to undo a permutation. It is
    NOT enough for rREST2, because `_apply_reservoir` REPLACES an entry in
    `state["configurations"]` outright: the refreshed walker's coordinates become the reservoir
    sample. Evaluating the CV from that list afterwards produced a row labelled "pre-exchange"
    whose value was measured on a configuration the run never propagated at that state -- it came
    out of the reservoir file.

    Every number involved is a torsion of the right molecule in the right range, so nothing in the
    output distinguishes the two. The only test that can is one that computes BOTH candidate
    values independently and asserts which of them the row holds.

PLATFORM_POLICY_EXEMPTION: this file establishes the SEMANTICS under `--cpu`, where a
deterministic refresh is cheap to force and the comparison is exact. The same semantics are
asserted on real CUDA in `test_cv_cuda_lanes.py`; what is checked here -- which of two candidate
coordinates a row was measured on -- is platform-independent bookkeeping.
"""
from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"

pytestmark = pytest.mark.slow

QUARTET = [4, 6, 8, 14]
CV_YAML = f"""\
schema_version: 1
collective_variables:
  - name: phi
    type: torsion
    atom_indices: {QUARTET}
"""

STATES = 3
EXCHANGE_EVERY = 10
EXCHANGES = 4
CV_EVERY = 5


def _build(root: Path):
    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
               "-log", "build/built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr
    (root / "cv.yaml").write_text(CV_YAML, encoding="utf-8")


def _reservoir(root: Path, *, frames: int = 6, tau_max: float = 0.5):
    """A phase-space reservoir whose frames sit at DELIBERATELY DISTINCT torsions.

    Each frame rotates the phi quartet's terminal atom about the central bond by a further 40
    degrees, so no reservoir sample can be confused with another or with anything the short
    ladder reaches by propagation. That separation is what makes "which coordinate was this row
    measured on" a decidable question rather than a numerical coincidence.
    """
    import mdtraj
    import numpy as np

    from md_tools.md.phase_space import PhaseSpaceWriter
    from md_tools.remd.generated import solute_document
    from md_tools.rest2 import identity as hamiltonian_identity
    from md_tools.run.preflight import check_scaling_plan, load_inputs

    loaded = load_inputs(str(root / "build" / "built.pdb"), str(root / "build" / "built.xml"))
    document = solute_document(loaded.pdb.topology, loaded.system, route="peptide")
    span = document.get("solute_atom_range")
    if span and document.get("solute_atom_indices_are_contiguous", False):
        indices = list(range(int(span[0]), int(span[1]) + 1))
    else:
        indices = list(range(int(document["n_solute_atoms"])))
    excluded = [tuple(int(a) for a in pair)
                for pair in (document.get("rest2") or {}).get("unscaled_central_bonds", [])]
    _audit, top = check_scaling_plan(loaded, solute_indices=indices, excluded_bonds=excluded,
                                     tau=tau_max, where="reservoir fixture")

    base = mdtraj.load(str(root / "build" / "built.pdb")).xyz[0]

    def _rotated(positions, degrees):
        """`positions` with the phi quartet's terminal atom rotated about the j-k axis.

        A displacement in an arbitrary direction barely moves a dihedral -- measured, an 0.08 nm
        shove changed phi by 0.14 degrees, which is not separable from the propagated value.
        Rotating about the CENTRAL BOND changes phi by exactly the rotation angle and leaves the
        terminal atom's distance to that bond unchanged by construction.
        """
        _i, j, k, l = QUARTET
        moved = np.array(positions, dtype=float)
        axis = moved[k] - moved[j]
        axis = axis / np.linalg.norm(axis)
        offset = moved[l] - moved[k]
        theta = math.radians(degrees)
        rotated = (offset * math.cos(theta)
                   + np.cross(axis, offset) * math.sin(theta)
                   + axis * float(np.dot(axis, offset)) * (1.0 - math.cos(theta)))
        moved[l] = moved[k] + rotated
        return moved

    rng = np.random.default_rng(20260904)
    writer = PhaseSpaceWriter(
        root / "reservoir.nc", n_atoms=base.shape[0], periodic=False,
        identity={"hamiltonian": hamiltonian_identity.identity_record(
            top, tau=tau_max, temperature_k=300.0, ensemble="NVT",
            solute_indices=indices, excluded_bonds=excluded)})
    try:
        for frame in range(frames):
            moved = _rotated(base, 40.0 * (frame + 1))
            # REAL momenta. Identically-zero velocities are refused by the source validator as a
            # configuration-only frame rather than a phase-space sample -- correctly, since the
            # probability-one rule is stated for a point in phase space and a reservoir of
            # motionless configurations is not one.
            velocities = rng.normal(scale=0.3, size=moved.shape)
            writer.append(positions=moved, velocities=velocities, box=None,
                          step=frame * 5, time_ps=float(frame))
    finally:
        writer.close()


@pytest.fixture(scope="module", params=["stored", "inherit"])
def project(request, tmp_path_factory):
    """An rREST2 project refreshing the top rung at EVERY exchange, under both velocity policies.

    `velocities: inherit` installs the recorded momentum ("stored"); `resample` draws fresh
    Maxwell momenta. Both go through the same configuration-replacement path, which is the path
    under test, so both are exercised.
    """
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    policy = {"stored": "inherit", "maxwell": "resample"}.get(request.param, request.param)
    root = tmp_path_factory.mktemp(f"rrest2-cv-{request.param}")
    _build(root)
    _reservoir(root)
    (root / "rREST2.config").write_text(yaml.safe_dump({
        "protocol": "rREST2", "solvent": "implicit",
        "stages": {"minimization_iterations": 5, "restrained_nvt_steps": 0,
                   "restrained_npt_steps": 0, "unrestrained_npt_steps": 0,
                   "production_steps": 0},
        "rest2": {"number_of_replicas": STATES, "exchange_interval_steps": EXCHANGE_EVERY,
                  "number_of_exchanges": EXCHANGES},
        # EVERY exchange refreshes, so the event is deterministic rather than hoped for.
        "reservoir": {"enabled": True, "path": "../reservoir.nc",
                      "refresh_interval_exchanges": 1, "velocities": policy},
        "reporting": {"crd_printout_solute": EXCHANGE_EVERY, "info_printout": EXCHANGE_EVERY,
                      "checkpoint_printout": EXCHANGE_EVERY},
        "collective_variables": {"file": str(root / "cv.yaml"), "interval_steps": CV_EVERY},
        "dynamics": {"seed": 20260904},
    }), encoding="utf-8")
    done = subprocess.run(
        CLI + ["build-md", "-odir", "./rREST2-run1", "--config", str(root / "rREST2.config")],
        cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr

    import openmm
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    pdb = PDBFile(str(root / "build" / "built.pdb"))
    system = XmlSerializer.deserialize((root / "build" / "built.xml").read_text(encoding="utf-8"))
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 1)
    (root / "initial_state.xml").write_text(XmlSerializer.serialize(
        context.getState(getPositions=True, getVelocities=True)), encoding="utf-8")
    return root


def _run(project: Path, destination: Path, *extra, expect=0):
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    user = project / "user.config"
    user.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(user)
    # DETERMINISM, not a tolerance. OpenMM's CPU platform sums its force reductions in
    # thread-completion order, so the same ladder with the same pinned seed follows a slightly
    # different trajectory when the pool size or the machine load changes. The assertions below
    # separate a propagated configuration from the reservoir sample it was refreshed from by
    # comparing torsions, and that separation depends on how far the walker actually drifted --
    # so an unpinned pool made this file's outcome depend on what else the machine was doing.
    #
    # It passed in isolation and failed inside a loaded full-suite run, where the drift came out
    # at 0.589 degrees against a threshold of 1.0. The threshold is not the problem and is not
    # touched; the run being irreproducible is.
    base["OPENMM_CPU_THREADS"] = "1"
    done = subprocess.run(
        [sys.executable, str(project / "rREST2-run1" / "rREST2.py"),
         "-p", str(project / "build" / "built.pdb"), "-s", str(project / "build" / "built.xml"),
         "-c", str(project / "initial_state.xml"),
         "-odir", str(destination), "--cpu", *extra],
        cwd=project / "rREST2-run1", capture_output=True, text=True, timeout=1800, env=base)
    if expect is not None:
        assert (done.returncode == 0) == (expect == 0), \
            done.stdout[-4000:] + done.stderr[-4000:]
    return done


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def completed(project, tmp_path_factory):
    destination = tmp_path_factory.mktemp("rrest2-cv-run") / "run"
    _run(project, destination)
    return destination


def _refresh_events(destination: Path):
    """(exchange_index, state, reservoir_frame) for every accepted refresh the run recorded."""
    from md_tools.remd import storage

    reporter = storage.ReplicaReporter(destination / "rREST2.nc", mode="r")
    try:
        events = reporter.reservoir_events()
    finally:
        reporter.close()
    return [(index, int(row[0]), int(row[1]))
            for index, row in enumerate(events) if int(row[3]) == 1 and int(row[0]) >= 0]


def _torsion(positions):
    from md_tools.cv import torsion_degrees

    return torsion_degrees(positions, QUARTET)


def test_a_refresh_actually_happened(completed):
    """Every assertion below is vacuous without one, so this states it plainly."""
    events = _refresh_events(completed)
    assert events, (
        "no accepted reservoir refresh was recorded, so the pre-refresh semantics were never "
        "exercised and the tests below prove nothing")


def test_the_row_holds_the_propagated_configuration_not_the_reservoir_sample(completed, project):
    """THE regression. Two candidate coordinates; the row must hold the propagated one.

    The reservoir sample's torsion is computed independently from the reservoir file. If the row
    matched THAT, the CV would have been evaluated after the replacement -- which is precisely the
    defect: a row labelled pre-exchange carrying a coordinate the state never propagated.
    """
    from md_tools.md.phase_space import PhaseSpaceReader

    events = _refresh_events(completed)
    assert events, "no refresh to check"

    with PhaseSpaceReader(project / "reservoir.nc") as reader:
        samples = {index: _torsion(reader.frame(index)[0])
                   for index in {frame for _e, _s, frame in events}}

    checked = 0
    for exchange_index, state_index, frame in events:
        step = (exchange_index + 1) * EXCHANGE_EVERY
        rows = {int(r["step"]): r for r in _rows(completed / f"cv_state{state_index}.csv")}
        if step not in rows:
            continue
        reported = float(rows[step]["phi"])
        sample = samples[frame]
        difference = abs((reported - sample + 180.0) % 360.0 - 180.0)
        assert difference > 1.0, (
            f"state {state_index} step {step} reports {reported}, and the reservoir sample it was "
            f"refreshed from has {sample}. The row was evaluated AFTER the replacement, so a row "
            f"labelled pre-exchange carries a coordinate this state never propagated")
        checked += 1
    assert checked, "no refreshed state had a CV row at its refresh step"


def test_the_refreshed_state_names_no_trajectory_frame(completed):
    """The frame written at that step holds the reservoir sample; this row does not.

    The walker index does not change under a refresh, so the swap rule alone would have named the
    frame. A refresh is a second, independent reason for the field to be empty.
    """
    events = _refresh_events(completed)
    assert events, "no refresh to check"
    checked = 0
    for exchange_index, state_index, _frame in events:
        step = (exchange_index + 1) * EXCHANGE_EVERY
        rows = {int(r["step"]): r for r in _rows(completed / f"cv_state{state_index}.csv")}
        if step not in rows:
            continue
        named = rows[step]["trajectory_frame_index"]
        assert named == "", (
            f"state {state_index} step {step} names frame {named!r}, but that frame holds the "
            f"reservoir sample and this row holds the propagated configuration")
        assert named != "-1"
        checked += 1
    assert checked, "no refreshed state had a CV row at its refresh step"


def test_unaffected_states_still_name_their_frames_correctly(completed, project):
    """The fix must not simply blank the column: untouched states keep working references."""
    mdtraj = pytest.importorskip("mdtraj")

    events = _refresh_events(completed)
    refreshed_at = {((e + 1) * EXCHANGE_EVERY, s) for e, s, _f in events}

    checked = 0
    for state_index in range(STATES):
        frames = mdtraj.load(str(completed / f"whole_state{state_index}_prod1.nc"),
                             top=str(project / "build" / "built.pdb"))
        for row in _rows(completed / f"cv_state{state_index}.csv"):
            named = row["trajectory_frame_index"]
            if named == "":
                continue
            assert (int(row["step"]), state_index) not in refreshed_at
            expected = math.degrees(float(
                mdtraj.compute_dihedrals(frames[int(named)], [QUARTET])[0][0]))
            reported = float(row["phi"])
            assert abs((reported - expected + 180.0) % 360.0 - 180.0) < 1e-2, (
                f"state {state_index} step {row['step']} names frame {named} but was measured "
                f"elsewhere")
            checked += 1
    assert checked, "no state retained a frame reference, so the fix over-blanked the column"
