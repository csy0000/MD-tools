"""One Amber NetCDF trajectory per fixed thermodynamic state.

The filename carries the state INDEX and never tau: `remd2.nc` means state 2, and which tau that
state holds belongs in metadata that can be validated rather than in a path anyone can rename.

"Amber NetCDF" is a convention, not a file extension. These tests do not assert on our own bytes;
they hand the result to the installed cpptraj and require it to say the file is an Amber
trajectory, and to concatenate a parent and an extension without a duplicated or missing frame at
the boundary.

PLATFORM_POLICY_EXEMPTION: file-format tests on a five-atom synthetic system. No dynamics.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "md_templates" / "openmm" / "templates"
netCDF4 = pytest.importorskip("netCDF4")
sys.path.insert(0, str(TEMPLATES))

import amber_trajectory as amber                                   # noqa: E402

CPPTRAJ = Path("/path/to/software/md-stack/conda/ambertools26/bin/cpptraj")
BOX = np.eye(3) * 2.5


def _topology(path, n_atoms=5):
    lines = [f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    "
             f"{i * 3.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
             for i in range(n_atoms)]
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def _write(path, times, *, state_index=0, n_atoms=5, periodic=True, tau=0.0):
    with amber.AmberTrajectoryWriter(path, n_atoms=n_atoms, state_index=state_index, tau=tau,
                                     temperature_k=300.0, periodic=periodic) as writer:
        for time_ps in times:
            writer.append(np.full((n_atoms, 3), time_ps * 0.01), time_ps=time_ps,
                          box_nm=BOX if periodic else None)
    return path


# --- names carry the state index, never tau ----------------------------------------------------

def test_the_name_is_the_state_index():
    assert amber.state_trajectory_name(0) == "remd0.nc"
    assert amber.state_trajectory_name(12) == "remd12.nc"
    assert amber.state_index_from_name("remd12.nc") == 12
    assert amber.state_index_from_name("/somewhere/remd3.nc") == 3


@pytest.mark.parametrize("tau", [0.0, 0.1, 0.5])
def test_the_name_does_not_depend_on_tau(tau, tmp_path):
    """Two ladders with different tau at the same index produce the same filename."""
    assert amber.state_trajectory_name(2) == "remd2.nc"
    path = _write(tmp_path / amber.state_trajectory_name(2), [1.0], state_index=2, tau=tau)
    assert path.name == "remd2.nc"
    assert amber.read_frames(path)["tau"] == pytest.approx(tau), (
        "tau is recorded in metadata, where it can be checked")


def test_a_name_without_a_numeric_index_is_refused():
    for bad in ("remd.nc", "remd_tau0p5.nc", "state2.nc", "remd2.dcd"):
        with pytest.raises(ValueError):
            amber.state_index_from_name(bad)


# --- the groupfile must name exactly one output per state ---------------------------------------

def test_one_contiguous_output_per_state_is_accepted():
    assert amber.validate_state_outputs(
        ["remd0.nc", "remd1.nc", "remd2.nc"]) == [0, 1, 2]


def test_a_duplicate_state_output_is_refused():
    with pytest.raises(ValueError, match="more than once"):
        amber.validate_state_outputs(["remd0.nc", "remd1.nc", "remd1.nc"])


def test_a_gap_in_the_state_outputs_is_refused():
    with pytest.raises(ValueError, match="contiguous"):
        amber.validate_state_outputs(["remd0.nc", "remd2.nc"])


def test_out_of_order_state_outputs_are_refused():
    """Position in the group file IS the state index, so order is meaning, not presentation."""
    with pytest.raises(ValueError, match="out of order"):
        amber.validate_state_outputs(["remd1.nc", "remd0.nc"])


# --- the Amber convention -----------------------------------------------------------------------

def test_the_file_carries_the_amber_convention(tmp_path):
    path = _write(tmp_path / "remd0.nc", [2.0, 4.0])
    with netCDF4.Dataset(str(path)) as d:
        assert d.Conventions == "AMBER"
        assert d.ConventionVersion == "1.0"
        assert set(("frame", "spatial", "atom")) <= set(d.dimensions)
        assert d.variables["coordinates"].units == "angstrom"
        assert d.variables["time"].units == "picosecond"
        assert d.variables["cell_lengths"].units == "angstrom"
        assert d.variables["cell_angles"].units == "degree"
        assert b"".join(d.variables["spatial"][:]).decode() == "xyz"


def test_positions_are_written_in_angstrom(tmp_path):
    """OpenMM works in nanometre; the conversion happens once, at this boundary."""
    path = tmp_path / "remd0.nc"
    with amber.AmberTrajectoryWriter(path, n_atoms=2, state_index=0, tau=0.0,
                                     temperature_k=300.0, periodic=False) as writer:
        writer.append(np.array([[0.0, 0.1, 0.2], [1.0, 0.0, 0.0]]), time_ps=1.0)
    with netCDF4.Dataset(str(path)) as d:
        got = np.array(d.variables["coordinates"][0], dtype=float)
    assert got == pytest.approx(np.array([[0.0, 1.0, 2.0], [10.0, 0.0, 0.0]]), abs=1e-4)


def test_a_periodic_trajectory_refuses_a_frame_with_no_box(tmp_path):
    with amber.AmberTrajectoryWriter(tmp_path / "remd0.nc", n_atoms=2, state_index=0, tau=0.0,
                                     temperature_k=300.0, periodic=True) as writer:
        with pytest.raises(ValueError, match="undefined density"):
            writer.append(np.zeros((2, 3)), time_ps=1.0)


def test_a_wrong_atom_count_is_refused(tmp_path):
    with amber.AmberTrajectoryWriter(tmp_path / "remd0.nc", n_atoms=5, state_index=0, tau=0.0,
                                     temperature_k=300.0, periodic=False) as writer:
        with pytest.raises(ValueError, match="shape"):
            writer.append(np.zeros((4, 3)), time_ps=1.0)


# --- the decisive tests: the real parser --------------------------------------------------------

@pytest.mark.skipif(not CPPTRAJ.is_file(), reason="AmberTools26 cpptraj is not installed")
def test_cpptraj_reads_it_as_an_amber_trajectory(tmp_path):
    _topology(tmp_path / "top.pdb")
    _write(tmp_path / "remd0.nc", [2.0, 4.0, 6.0, 8.0])
    (tmp_path / "in").write_text("parm top.pdb\ntrajin remd0.nc\nrun\n")
    done = subprocess.run([str(CPPTRAJ), "-i", "in"], capture_output=True, text=True,
                          cwd=tmp_path)
    out = done.stdout + done.stderr
    assert "AMBER trajectory" in out, out
    assert "reading 4 of 4" in out, out
    assert "Error" not in out, out


@pytest.mark.skipif(not CPPTRAJ.is_file(), reason="AmberTools26 cpptraj is not installed")
def test_cpptraj_concatenates_a_parent_and_an_extension_without_a_boundary_seam(tmp_path):
    """What the 5 ns + 5 ns example needs: the extension holds only its own segment, its times are
    absolute and continue after the parent's, and joining them repeats nothing and drops nothing."""
    _topology(tmp_path / "top.pdb")
    _write(tmp_path / "parent.nc", [2.0, 4.0, 6.0, 8.0])
    _write(tmp_path / "ext.nc", [10.0, 12.0])
    (tmp_path / "in").write_text(
        "parm top.pdb\ntrajin parent.nc\ntrajin ext.nc\ntrajout joined.nc netcdf\nrun\n")
    done = subprocess.run([str(CPPTRAJ), "-i", "in"], capture_output=True, text=True,
                          cwd=tmp_path)
    assert "Error" not in done.stdout + done.stderr, done.stdout + done.stderr

    joined = amber.read_frames(tmp_path / "joined.nc")
    assert joined["n_frames"] == 6
    assert joined["times_ps"] == pytest.approx([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]), (
        "a duplicated boundary frame would show as a repeated time; a missing one as a gap")


@pytest.mark.skipif(not CPPTRAJ.is_file(), reason="AmberTools26 cpptraj is not installed")
def test_each_state_gets_its_own_readable_trajectory(tmp_path):
    _topology(tmp_path / "top.pdb")
    for state in range(3):
        _write(tmp_path / amber.state_trajectory_name(state), [2.0, 4.0], state_index=state,
               tau=0.1 * state)
    for state in range(3):
        name = amber.state_trajectory_name(state)
        (tmp_path / "in").write_text(f"parm top.pdb\ntrajin {name}\nrun\n")
        done = subprocess.run([str(CPPTRAJ), "-i", "in"], capture_output=True, text=True,
                              cwd=tmp_path)
        assert "AMBER trajectory" in done.stdout + done.stderr
        assert amber.read_frames(tmp_path / name)["state_index"] == state
