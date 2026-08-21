"""DCD tail recovery must understand real OpenMM records.

Finding 3. The previous truncation treated a frame as three coordinate blocks. A *periodic* OpenMM
frame also carries a 48-byte unit cell inside Fortran markers -- 56 bytes -- so the old arithmetic
landed 56 bytes short per frame and corrupted every explicit trajectory it touched. Measured on a
6-atom periodic DCD: old arithmetic 96 bytes per frame, real size 152.

The flag cannot be inferred from the solvent mode: `DCDReporter(..., atomSubset=...)` takes its box
vectors from the topology, so a selected-atom trajectory written with `enforcePeriodicBox=False`
still carries a unit cell when the topology has one. That case is tested explicitly.

## The independent reader

`_read_dcd` below is a second, deliberately separate parser. It shares no code with `dcdtail`, so a
misunderstanding of the format in the implementation cannot be reproduced by the checker and hide
the fault. mdtraj and MDAnalysis are not installed and the instruction forbids adding a dependency
to avoid understanding the format, so the checker is written out here.
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------------------------
# an independent DCD reader, sharing nothing with the implementation under test
# ---------------------------------------------------------------------------------------------

def _read_dcd(path: Path) -> dict:
    """Parse a DCD from first principles and return its frames, box records and header fields."""
    data = Path(path).read_bytes()
    if struct.unpack_from("<i", data, 0)[0] != 84 or data[4:8] != b"CORD":
        raise AssertionError("not a little-endian OpenMM DCD")
    n_frames = struct.unpack_from("<i", data, 8)[0]
    first_step = struct.unpack_from("<i", data, 12)[0]
    interval = struct.unpack_from("<i", data, 16)[0]
    last_step = struct.unpack_from("<i", data, 20)[0]
    box_flag = struct.unpack_from("<i", data, 48)[0]

    cursor = 92                                          # past the control record
    title_bytes = struct.unpack_from("<i", data, cursor)[0]
    cursor += 4 + title_bytes + 4
    assert struct.unpack_from("<i", data, cursor)[0] == 4
    n_atoms = struct.unpack_from("<i", data, cursor + 4)[0]
    cursor += 12

    frames, boxes = [], []
    while cursor < len(data):
        if box_flag:
            assert struct.unpack_from("<i", data, cursor)[0] == 48
            boxes.append(struct.unpack_from("<6d", data, cursor + 4))
            cursor += 56
        coordinates = []
        for _ in range(3):
            payload = struct.unpack_from("<i", data, cursor)[0]
            assert payload == 4 * n_atoms, f"block declares {payload} for {n_atoms} atoms"
            coordinates.append(np.frombuffer(data, dtype="<f4", count=n_atoms, offset=cursor + 4))
            cursor += 4 + payload + 4
        frames.append(np.stack(coordinates, axis=1))
    return {"n_frames_header": n_frames, "n_frames_actual": len(frames), "n_atoms": n_atoms,
            "periodic": bool(box_flag), "first_step": first_step, "interval": interval,
            "last_step": last_step, "frames": frames, "boxes": boxes}


# ---------------------------------------------------------------------------------------------
# fixtures that write real OpenMM DCDs
# ---------------------------------------------------------------------------------------------

def _write_dcd(path: Path, *, periodic: bool, frames: int, n_atoms: int = 6, subset=None):
    import openmm
    from openmm import app, unit

    system = openmm.System()
    for _ in range(n_atoms):
        system.addParticle(12.0 * unit.dalton)
    nonbonded = openmm.NonbondedForce()
    for _ in range(n_atoms):
        nonbonded.addParticle(0.0, 0.3 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic if periodic
                                 else openmm.NonbondedForce.NoCutoff)
    if periodic:
        nonbonded.setCutoffDistance(0.5 * unit.nanometer)
    system.addForce(nonbonded)

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("LIG", chain)
    for i in range(n_atoms):
        topology.addAtom(f"C{i}", app.element.carbon, residue)
    if periodic:
        box = (np.eye(3) * 2.0).tolist()
        system.setDefaultPeriodicBoxVectors(*box)
        topology.setPeriodicBoxVectors(box)

    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                                 0.002 * unit.picosecond)
    simulation = app.Simulation(topology, system, integrator,
                                openmm.Platform.getPlatformByName("Reference"))
    simulation.context.setPositions(
        np.random.RandomState(3).rand(n_atoms, 3) * 0.5 * unit.nanometer)
    options = {"enforcePeriodicBox": periodic}
    if subset is not None:
        options["atomSubset"] = subset
    simulation.reporters.append(app.DCDReporter(str(path), 2, **options))
    simulation.step(2 * frames)
    for reporter in simulation.reporters:
        handle = getattr(reporter, "_out", None)
        if handle is not None:
            handle.close()
    simulation.reporters.clear()
    del simulation, integrator


CASES = [
    pytest.param(True, None, id="explicit-periodic-all-atom"),
    pytest.param(True, [0, 1, 2], id="explicit-periodic-subset-unwrapped"),
    pytest.param(False, None, id="implicit-nonperiodic-all-atom"),
    pytest.param(False, [0, 1, 2], id="implicit-nonperiodic-subset"),
]


# ---------------------------------------------------------------------------------------------
# the scanner
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("periodic,subset", CASES)
def test_the_scanner_agrees_with_an_independent_reader(tmp_path, periodic, subset):
    from md_templates.openmm.dcdtail import frame_offsets

    path = tmp_path / "traj.dcd"
    _write_dcd(path, periodic=periodic, frames=5, subset=subset)

    scanned = frame_offsets(path)
    reference = _read_dcd(path)
    assert scanned["complete_frames"] == reference["n_frames_actual"] == 5
    assert scanned["n_atoms"] == reference["n_atoms"] == (len(subset) if subset else 6)
    assert scanned["periodic"] is reference["periodic"]
    assert scanned["trailing_bytes"] == 0


def test_a_periodic_subset_trajectory_still_carries_a_unit_cell(tmp_path):
    """The case that made coordinate-only arithmetic wrong in a way solvent mode cannot predict."""
    from md_templates.openmm.dcdtail import frame_offsets

    path = tmp_path / "subset.dcd"
    _write_dcd(path, periodic=True, frames=4, subset=[0, 1, 2])
    scanned = frame_offsets(path)
    assert scanned["periodic"] is True, "the subset DCD dropped its unit cell"
    assert scanned["n_atoms"] == 3

    coordinate_only = 3 * (4 + 4 * 3 + 4)
    real = 56 + 3 * (8 + 4 * 3)
    assert real - coordinate_only == 56, "the unit-cell record is the 56 bytes the old code missed"


# ---------------------------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("periodic,subset", CASES)
@pytest.mark.parametrize("tail", [1, 3])
def test_recovery_removes_only_the_tail_and_stays_readable(tmp_path, periodic, subset, tail):
    """Committed frames must survive byte-for-byte, and the header must stay self-consistent."""
    from md_templates.openmm.dcdtail import truncate_to_frames

    path = tmp_path / "traj.dcd"
    _write_dcd(path, periodic=periodic, frames=5 + tail, subset=subset)
    before = _read_dcd(path)

    result = truncate_to_frames(path, 5)
    assert result["action"] == "truncated"

    after = _read_dcd(path)
    assert after["n_frames_header"] == 5, "the header frame count was not updated"
    assert after["n_frames_actual"] == 5, "the file does not hold exactly five frames"
    assert after["n_atoms"] == before["n_atoms"]
    assert after["periodic"] == before["periodic"]
    # the two header fields must agree: last step follows from first step and interval
    assert after["last_step"] == after["first_step"] + (5 - 1) * after["interval"]

    for index in range(5):
        assert np.array_equal(after["frames"][index], before["frames"][index]), (
            f"committed frame {index} changed during recovery")
    if periodic:
        assert after["boxes"][:5] == before["boxes"][:5]


def test_an_incomplete_final_record_is_treated_as_a_partial_frame(tmp_path):
    """A writer that died mid-frame leaves bytes that are not a frame. They are not counted."""
    from md_templates.openmm.dcdtail import frame_offsets, truncate_to_frames

    path = tmp_path / "torn.dcd"
    _write_dcd(path, periodic=True, frames=4)
    with path.open("ab") as handle:                       # half of a coordinate block
        handle.write(struct.pack("<i", 4 * 6))
        handle.write(b"\x00" * 10)

    scanned = frame_offsets(path)
    assert scanned["complete_frames"] == 4
    assert scanned["trailing_bytes"] > 0, "the torn bytes were absorbed into a frame"

    truncate_to_frames(path, 4)
    after = _read_dcd(path)
    assert after["n_frames_actual"] == 4 and after["n_frames_header"] == 4


def test_keeping_more_frames_than_exist_is_refused(tmp_path):
    """Shorter than the watermark is missing history, which Finding 4 must refuse, not pad."""
    from md_templates.openmm.dcdtail import DCDFormatError, truncate_to_frames

    path = tmp_path / "short.dcd"
    _write_dcd(path, periodic=False, frames=2)
    with pytest.raises(DCDFormatError, match="missing history"):
        truncate_to_frames(path, 5)


def test_a_big_endian_or_non_dcd_file_is_refused(tmp_path):
    """Editing a file this parser does not understand would corrupt it silently."""
    from md_templates.openmm.dcdtail import DCDFormatError, read_header

    big_endian = tmp_path / "be.dcd"
    big_endian.write_bytes(struct.pack(">i", 84) + b"CORD" + b"\x00" * 200)
    with pytest.raises(DCDFormatError, match="big-endian"):
        read_header(big_endian)

    junk = tmp_path / "junk.dcd"
    junk.write_bytes(b"not a dcd at all" * 8)
    with pytest.raises(DCDFormatError):
        read_header(junk)


def test_truncation_is_atomic_and_leaves_no_temporary_behind(tmp_path):
    from md_templates.openmm.dcdtail import truncate_to_frames

    path = tmp_path / "traj.dcd"
    _write_dcd(path, periodic=True, frames=6)
    truncate_to_frames(path, 2)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "traj.dcd"]
    assert not leftovers, f"recovery left temporary files behind: {leftovers}"
    assert _read_dcd(path)["n_frames_actual"] == 2
