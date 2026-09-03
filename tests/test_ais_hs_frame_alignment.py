"""The potentials on an HS row belong to the coordinate that row names. Verified from the files.

THE DEFECT THIS EXISTS FOR

A switching update moves the parameters at frozen `x_j`, then propagates to `x_{j+1}`. The
previous schema wrote the trajectory frame for `x_{j+1}` and, on the same row, the potential basis
measured at `x_j` -- one propagation earlier -- under column names that read as potentials at that
frame. A Hummer-Szabo reweighting from that file pairs the work of one configuration with the
energy of another. Nothing raises. Every number is plausible.

HOW THIS TESTS IT

Not by reading the runtime's own numbers back and checking they are self-consistent -- they were
self-consistent before, and wrong. Instead:

  1. run a real AIS path and let it write real `AIS_trajNNNN.nc` files and real CSVs;
  2. REOPEN the trajectory and read the coordinates of the exact frame a row names;
  3. build a FRESH OpenMM Context, put those coordinates in it, set the row's tau;
  4. recompute the three basis groups and the direct potential from scratch;
  5. compare with the row.

Step 3 is the point. The recomputation shares no state with the run: a different Context, built
after the fact, from coordinates that came off the disk. If the row's potentials belonged to a
different coordinate, these numbers would not match, and under the old schema they did not.

The identity checked is

    U(tau, x) = U_non_scaled(x) + sqrt(lambda) U_sqrt_scaled(x) + lambda U_lin_scaled(x)

with `a = 1 - tau` and `lambda = a^2`, so `sqrt(lambda) = a`.

PLATFORM_POLICY_EXEMPTION: these run on whatever `machine.openmm.platform` resolves to. What is
under test is which COORDINATE a number was measured at, which is identical on every platform. The
same recomputation is performed on real CUDA, for implicit and explicit PME alike, in
`tests/test_cuda_coverage_matrix.py`.
"""
from __future__ import annotations

import csv
import json
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

#: Five DIFFERENT divisors of `switching_steps`: a switch every 2 steps, a work observation every
#: 6, a trajectory frame every 10, a system row every 20 and a checkpoint every 15.
#:
#: The observation cadence is a multiple of the switching cadence because the schedule requires it
#: -- observations that did not land on the parameter-update grid would be evenly spaced only
#: after rounding, and that refusal is correct. Nothing else here shares a factor it does not have
#: to.
#:
#: The consequences are all deliberate. Observations fall on multiples of 6 and frames on
#: multiples of 10, so only multiples of 30 are both: 5 of the 21 observation rows carry
#: potentials and 16 do not, and 8 of the 13 frames have no observation row naming them. That is
#: exactly the case a schema which silently borrowed a neighbouring frame would have hidden, and
#: here it would have been wrong on three quarters of the rows.
SWITCHING_STEPS = 120
UPDATE_EVERY = 2
OBSERVE_EVERY = 6
FRAME_EVERY = 10
STATE_EVERY = 20
CHECKPOINT_EVERY = 15


def frame_roundtrip_tolerance(energy: float) -> float:
    """How far a recomputation from a SAVED frame may sit from the value recorded during the run.

    This is not the decomposition's tolerance and it is not the platform's. It is the cost of the
    round trip through the file: AMBER NetCDF stores coordinates as float32 in angstrom, so a
    frame read back differs from the in-memory double coordinates by ~1e-7 relative, and a
    potential evaluated at the perturbed coordinates differs accordingly.

    It is a real limit, not an artefact of the test -- a downstream Hummer-Szabo consumer reading
    the same file sees exactly this. Which is the argument for measuring it rather than tightening
    it: the number below is what the format costs, and the assertion is that the misalignment
    being tested for is orders of magnitude larger than that.

    Scaled by the TOTAL energy of the system, not by the individual component being compared.
    That distinction matters and is easy to get wrong: each basis value is a linear combination of
    three probe energies, each of order the total potential, with fit coefficients of magnitude 1,
    3 and 4. So the absolute error on a small component is set by the size of the LARGEST energy
    that went into it -- a solvated `sqrt_scaled` of -9 kJ/mol inherits the round-trip error of a
    -1.6e4 kJ/mol total, and a bound proportional to 9 would fail for a reason that has nothing to
    do with the alignment under test.
    """
    return max(abs(float(energy)) * 5.0e-5, 1.0e-3)


def _environment(work: Path, **extra) -> dict[str, str]:
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])])
    config = work / "user.config"
    config.write_text(yaml.safe_dump(
        {"schema_version": "1.0", "user": {"person_id": "t", "name": "T"}}), encoding="utf-8")
    base["MD_TOOLS_CONFIG"] = str(config)
    base.update(extra)
    return base


@pytest.fixture(scope="module")
def implicit_run(tmp_path_factory):
    """A real implicit AIS run with five distinct cadences. Built and run once."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("hs-implicit")

    (root / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout + built.stderr

    import mdtraj

    frames = mdtraj.load(str(root / "built.pdb"))
    mdtraj.join([frames] * 8).save_dcd(str(root / "source.dcd"))

    (root / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "implicit",
        "ais": {"number_of_paths": 2, "switching_steps": SWITCHING_STEPS,
                "observation_interval_steps": OBSERVE_EVERY,
                "parameter_update_interval_steps": UPDATE_EVERY},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"solute_printout": FRAME_EVERY, "system_printout": STATE_EVERY,
                      "checkpoint_printout": CHECKPOINT_EVERY}}), encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(root / "project"),
                                      "--config", str(root / "AIS.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = subprocess.run(
        [sys.executable, str(root / "project" / "AIS.py"),
         "-p", str(root / "built.pdb"), "-s", str(root / "built.xml"),
         "-source-traj", str(root / "source.dcd"), "-odir", str(root / "run"), "--cpu"],
        cwd=root, capture_output=True, text=True, timeout=3600, env=_environment(root))
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return root


# --- the independent recomputation ----------------------------------------------------------------

def recompute_at_frame(root: Path, run: Path, *, path_id: int, frame_index: int, tau: float):
    """Rebuild a Context from a SAVED frame and recompute the three groups and U(tau).

    Shares nothing with the run that produced the file: a fresh System, a fresh Context, and
    coordinates read off the disk. That independence is the whole point -- a row's numbers being
    consistent with the runtime's other numbers proves nothing about which coordinate they belong
    to, which is how the previous schema was self-consistent and wrong.
    """
    import mdtraj
    from openmm import Platform, VerletIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile

    from md_tools.ais.decomposition import BASIS_PROBE_AMPLITUDES, Components
    from md_tools.md.stage import solute_atom_indices
    from md_tools.openmm.system import classify_omega_bonds
    from md_tools.rest2.scaler import TauSwitcher

    pdb = PDBFile(str(root / "built.pdb"))
    base = XmlSerializer.deserialize((root / "built.xml").read_text(encoding="utf-8"))
    solute = solute_atom_indices(pdb.topology)
    omega = classify_omega_bonds(pdb.topology, solute, route="peptide", ligand_sdf=None)
    excluded = [tuple(int(a) for a in bond) for bond in omega.get("omega_unscaled_bonds", [])]

    switcher = TauSwitcher(base, solute, excluded)
    system = switcher.prepared_system(tau)
    context = Platform.getPlatformByName("Reference")
    from openmm import Context

    handle = Context(system, VerletIntegrator(0.001), context)

    trajectory = run / f"AIS_traj{path_id:04d}.nc"
    with mdtraj.formats.NetCDFTrajectoryFile(str(trajectory)) as reader:
        reader._frame_index = frame_index
        coordinates, _time, lengths, angles = reader.read(1)
    positions = coordinates[0] / 10.0                       # angstrom -> nanometre
    if lengths is not None:
        from openmm.app.internal.unitcell import computePeriodicBoxVectors
        import numpy

        radians = numpy.radians(angles[0])
        handle.setPeriodicBoxVectors(*computePeriodicBoxVectors(
            float(lengths[0][0]) / 10.0, float(lengths[0][1]) / 10.0,
            float(lengths[0][2]) / 10.0,
            float(radians[0]), float(radians[1]), float(radians[2])))
    handle.setPositions(positions * unit.nanometer)

    energies = []
    for amplitude in BASIS_PROBE_AMPLITUDES:
        switcher.set_amplitude(handle, system, amplitude)
        energies.append(handle.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole))
    components = Components.from_probe(BASIS_PROBE_AMPLITUDES, energies)

    switcher.set_tau(handle, system, tau)
    direct = handle.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    return components, direct


def _hs_rows(run: Path):
    with (run / "AIS_hs.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _observations(run: Path, path_id: int = 0):
    with (run / f"path_{path_id:04d}" / "observations.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


# --- the schedule is genuinely five different cadences --------------------------------------------

def test_the_schedule_really_does_use_five_different_cadences(implicit_run):
    """Guard on the fixture. Identical cadences would make every test below vacuous."""
    record = json.loads((implicit_run / "run" / "AIS_run.json").read_text(encoding="utf-8"))
    schedule = record["schedule"]
    cadences = {
        "switch": schedule["parameter_update_interval_steps"],
        "work": schedule["observation_interval_steps"],
        "trajectory": schedule["trajectory_interval_steps"],
        "system": schedule["state_interval_steps"],
        "checkpoint": schedule["checkpoint_interval_steps"],
    }
    assert len(set(cadences.values())) == 5, cadences
    for name, value in cadences.items():
        assert SWITCHING_STEPS % value == 0, f"{name}={value} does not divide {SWITCHING_STEPS}"


def test_rows_without_a_saved_coordinate_carry_no_potentials(implicit_run):
    """Half the observation rows have no frame. Their potential cells must be EMPTY.

    A schema that filled them from the nearest frame would look complete and be exactly the defect
    this file exists for -- and with these cadences it would be wrong on half the rows.
    """
    rows = _observations(implicit_run / "run")
    unaligned = [row for row in rows if row["coordinate_frame_index"] == ""]
    aligned = [row for row in rows if row["coordinate_frame_index"] != ""]
    assert unaligned, "these cadences should leave some observations without a frame"
    assert aligned, "these cadences should leave some observations with one"
    for row in unaligned:
        for name in ("potential_non_scaled_kj_mol", "potential_sqrt_scaled_kj_mol",
                     "potential_lin_scaled_kj_mol", "potential_reconstructed_kj_mol",
                     "potential_direct_kj_mol"):
            assert row[name] == "", (row["protocol_step"], name)
    # And the work columns are present on every row, aligned or not: work does not depend on a
    # coordinate having been saved.
    for row in rows:
        assert row["cumulative_work_kj_mol"] != ""
        assert row["total_work_sqrt_scaled_kj_mol"] != ""


def test_the_hs_table_holds_exactly_the_frame_aligned_rows(implicit_run):
    run = implicit_run / "run"
    aligned = [row for row in _observations(run) if row["coordinate_frame_index"] != ""]
    hs = [row for row in _hs_rows(run) if int(row["path_id"]) == 0]
    assert len(hs) == len(aligned), (len(hs), len(aligned))
    for row in hs:
        assert row["coordinate_frame_index"] != ""
        for name in ("potential_non_scaled_kj_mol", "potential_sqrt_scaled_kj_mol",
                     "potential_lin_scaled_kj_mol", "potential_reconstructed_kj_mol",
                     "potential_direct_kj_mol", "total_work_kj_mol", "tau"):
            assert row[name] != "", name


# --- the recomputation ----------------------------------------------------------------------------

def test_every_hs_row_matches_an_independent_recomputation_at_its_own_frame(implicit_run):
    """THE test. Reopen each named frame, rebuild a Context, recompute, compare.

    Covers observation zero, every interior frame and the final frame, because it walks the whole
    table rather than sampling it.
    """
    root, run = implicit_run, implicit_run / "run"
    rows = _hs_rows(run)
    assert rows, "no HS rows were written"

    worst = {"non_scaled": 0.0, "sqrt_scaled": 0.0, "lin_scaled": 0.0,
             "reconstructed": 0.0, "direct": 0.0}
    for row in rows:
        tau = float(row["tau"])
        components, direct = recompute_at_frame(
            root, run, path_id=int(row["path_id"]),
            frame_index=int(row["coordinate_frame_index"]), tau=tau)

        # One scale for the whole row: the total potential, which is what every probe energy
        # behind these components was of the order of.
        allowed = frame_roundtrip_tolerance(float(row["potential_direct_kj_mol"]))
        for group in ("non_scaled", "sqrt_scaled", "lin_scaled"):
            recorded = float(row[f"potential_{group}_kj_mol"])
            error = abs(recorded - getattr(components, group))
            worst[group] = max(worst[group], error)
            assert error < allowed, (
                f"path {row['path_id']} frame {row['coordinate_frame_index']}: {group} recorded "
                f"{recorded}, recomputed {getattr(components, group)}")

        error = abs(float(row["potential_reconstructed_kj_mol"]) - components.total_at(tau))
        worst["reconstructed"] = max(worst["reconstructed"], error)
        assert error < allowed

        error = abs(float(row["potential_direct_kj_mol"]) - direct)
        worst["direct"] = max(worst["direct"], error)
        assert error < allowed, (
            "the row's direct potential does not match a fresh evaluation at its own frame")

    print(f"\nmax |recorded - independently recomputed| over {len(rows)} HS rows: {worst}")


def test_the_potentials_are_not_those_of_the_previous_frame(implicit_run):
    """The regression, stated as its own test: the OLD behaviour must now FAIL.

    Recomputing at the frame BEFORE the one a row names must disagree with the row. Without this,
    every assertion above would still pass on a file whose potentials were uniformly shifted by
    one frame -- which is precisely what the previous schema wrote.
    """
    root, run = implicit_run, implicit_run / "run"
    rows = [row for row in _hs_rows(run) if int(row["coordinate_frame_index"]) > 0]
    assert rows, "need at least one row after the first frame"

    disagreements = 0
    for row in rows:
        tau = float(row["tau"])
        shifted, _direct = recompute_at_frame(
            root, run, path_id=int(row["path_id"]),
            frame_index=int(row["coordinate_frame_index"]) - 1, tau=tau)
        allowed = frame_roundtrip_tolerance(float(row["potential_direct_kj_mol"]))
        if abs(float(row["potential_lin_scaled_kj_mol"]) - shifted.lin_scaled) > allowed:
            disagreements += 1
    assert disagreements == len(rows), (
        f"{len(rows) - disagreements} of {len(rows)} rows agree with the PREVIOUS frame's "
        f"potentials as well as their own -- the frames are indistinguishable, so this test "
        f"cannot detect the misalignment it exists for")


def test_the_work_identity_holds_independently_at_every_switch(implicit_run):
    """`dW_total == dW_non + dW_sqrt + dW_lin` on every emitted work observation."""
    from md_tools.ais.decomposition import reconstruction_tolerance
    from md_tools.build.record import read_record

    # The precision the run ACTUALLY used, read from its own record rather than assumed. The CPU
    # platform is not double: its nonbonded sums are single-precision, so asserting a
    # double-precision bound here would be holding the run to a promise it never made.
    precision = (read_record(implicit_run / "run" / "AIS.log")["acceleration"].get(
        "cuda_precision") or "single")

    rows = _observations(implicit_run / "run")
    # Each emitted row covers `OBSERVE_EVERY / UPDATE_EVERY` switches, and each switch carries the
    # decomposition's own documented tolerance. The bound is that tolerance times the number of
    # switches in the row -- the runtime's own per-switch bound, accumulated, rather than a
    # tighter number invented here that the runtime never promised.
    per_row = OBSERVE_EVERY // UPDATE_EVERY
    worst_delta = worst_total = 0.0
    for row in rows:
        parts = sum(float(row[f"delta_work_{group}_kj_mol"])
                    for group in ("non_scaled", "sqrt_scaled", "lin_scaled"))
        measured = float(row["incremental_work_kj_mol"])
        allowed = reconstruction_tolerance(
            max(abs(measured), abs(float(row["cumulative_work_kj_mol"]))),
            precision=precision) * per_row
        worst_delta = max(worst_delta, abs(parts - measured))
        assert abs(parts - measured) <= max(allowed, 1e-4), row["protocol_step"]
        cumulative = sum(float(row[f"total_work_{group}_kj_mol"])
                         for group in ("non_scaled", "sqrt_scaled", "lin_scaled"))
        total = float(row["cumulative_work_kj_mol"])
        worst_total = max(worst_total, abs(cumulative - total))
        assert abs(cumulative - total) <= max(
            reconstruction_tolerance(total, precision=precision) * len(rows) * per_row,
            1e-4), (
            row["protocol_step"])
    print(f"\nmax |sum of components - measured work| at {precision} precision: "
          f"delta {worst_delta:.3e}, cumulative {worst_total:.3e} kJ/mol")
    # The non-scaled work is identically zero: that group has no tau dependence.
    assert all(float(row["delta_work_non_scaled_kj_mol"]) == 0.0 for row in rows)


def test_delta_work_is_the_sum_since_the_previous_observation_not_one_switch(implicit_run):
    """Documented, and checked: these cadences put two or three switches in every row.

    With a switch every 2 steps and an observation every 5, the rows cover 2 or 3 switches each --
    so `delta_work_kj_mol` is a SUM, and a reader who assumed one switch would be wrong by a
    factor that changes from row to row.
    """
    record = json.loads((implicit_run / "run" / "AIS_run.json").read_text(encoding="utf-8"))
    schedule = record["schedule"]
    assert schedule["observation_interval_steps"] != schedule["parameter_update_interval_steps"]

    from md_tools.ais.decomposition import DECOMPOSITION_SCHEMA

    assert "sum of every switch since the previous emitted work observation" in (
        DECOMPOSITION_SCHEMA["delta_work_meaning"])

    rows = _observations(implicit_run / "run")
    running = 0.0
    for row in rows[1:]:
        running += float(row["incremental_work_kj_mol"])
        assert abs(running - float(row["cumulative_work_kj_mol"])) < 1e-9, row["protocol_step"]


# --- interruption and resume ----------------------------------------------------------------------

def _run_ais(root: Path, destination: Path, *, environment=None, extra=()):
    base = _environment(root)
    base.update(environment or {})
    return subprocess.run(
        [sys.executable, str(root / "project" / "AIS.py"),
         "-p", str(root / "built.pdb"), "-s", str(root / "built.xml"),
         "-source-traj", str(root / "source.dcd"), "-odir", str(destination), "--cpu", *extra],
        cwd=root, capture_output=True, text=True, timeout=3600, env=base)


@pytest.mark.parametrize("boundary", ["after-checkpoint-write", "before-pointer-replace",
                                      "after-pointer-replace", "before-frame", "after-frame",
                                      "before-work-row", "after-work-row",
                                      "before-state-row", "after-state-row"])
def test_a_resumed_path_produces_no_duplicate_or_mismatched_hs_rows(boundary, implicit_run,
                                                                    tmp_path):
    """Crash at each stream and transaction boundary, resume, and check the HS table.

    Three ways a resume could corrupt this table, all of them silent:

      a DUPLICATE row, if the truncation missed one and the resume appended over it;
      a MISSING row, if the truncation cut one the checkpoint had committed;
      a MISMATCHED row, whose potentials belong to a frame index the resume renumbered.

    The third is the one this file exists for, and it is the one a count-based check would miss
    entirely -- so every row is recomputed at the frame it names, exactly as the uninterrupted
    test does.
    """
    destination = tmp_path / f"resume-{boundary}"
    destination.mkdir()

    crashed = _run_ais(implicit_run, destination,
                       environment={"MD_TOOLS_CHECKPOINT_FAULT": boundary,
                                    "MD_TOOLS_CHECKPOINT_FAULT_AFTER": "1"})
    assert crashed.returncode != 0, crashed.stdout[-2000:] + crashed.stderr[-2000:]

    resumed = _run_ais(implicit_run, destination, extra=["--resume"])
    assert resumed.returncode == 0, resumed.stdout[-4000:] + resumed.stderr[-4000:]

    rows = _hs_rows(destination)
    reference = _hs_rows(implicit_run / "run")
    assert len(rows) == len(reference), (
        f"{boundary}: {len(rows)} HS rows against the uninterrupted run's {len(reference)}")

    keys = [(row["path_id"], row["switch_step"]) for row in rows]
    assert len(keys) == len(set(keys)), f"{boundary}: a (path, step) pair appears twice"
    frames = [(row["path_id"], row["coordinate_frame_index"]) for row in rows]
    assert len(frames) == len(set(frames)), f"{boundary}: a frame is named by two rows"

    # And every row still describes the frame it names, after the truncation renumbered nothing.
    for row in rows:
        tau = float(row["tau"])
        components, direct = recompute_at_frame(
            implicit_run, destination, path_id=int(row["path_id"]),
            frame_index=int(row["coordinate_frame_index"]), tau=tau)
        recorded = float(row["potential_direct_kj_mol"])
        assert abs(recorded - direct) < frame_roundtrip_tolerance(recorded), (
            f"{boundary}: path {row['path_id']} frame {row['coordinate_frame_index']} does not "
            f"match a recomputation at that frame")


def test_a_resumed_run_reaches_the_same_final_work_as_an_uninterrupted_one(implicit_run,
                                                                           tmp_path):
    """Same seeds, same source, same schedule: the science must not depend on the interruption."""
    destination = tmp_path / "resume-work"
    destination.mkdir()
    crashed = _run_ais(implicit_run, destination,
                       environment={"MD_TOOLS_CHECKPOINT_FAULT": "after-pointer-replace",
                                    "MD_TOOLS_CHECKPOINT_FAULT_AFTER": "1"})
    assert crashed.returncode != 0
    resumed = _run_ais(implicit_run, destination, extra=["--resume"])
    assert resumed.returncode == 0, resumed.stdout[-4000:] + resumed.stderr[-4000:]

    def totals(run: Path):
        with (run / "AIS_paths.csv").open(newline="") as handle:
            return {row["path_index"]: float(row["total_work_kj_mol"])
                    for row in csv.DictReader(handle)}

    reference, recovered = totals(implicit_run / "run"), totals(destination)
    assert set(reference) == set(recovered)
    # A relative bound, at the precision the run recorded. `loadCheckpoint` restores the Context
    # exactly, but the CPU platform's nonbonded sums are single precision and its thread partition
    # need not be identical across two processes -- so the resumed path is the same trajectory to
    # within the platform's own arithmetic, not to the last bit. Demanding the last bit would be
    # asserting a determinism guarantee OpenMM does not make on this platform.
    for path_id, expected in reference.items():
        allowed = max(abs(expected) * 1.0e-5, 1.0e-4)
        assert abs(recovered[path_id] - expected) < allowed, (
            f"path {path_id}: resumed {recovered[path_id]} against uninterrupted {expected}")


# --- explicit PME ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def explicit_run(tmp_path_factory):
    """The same thing over a solvated box, where PME, the Ewald self-energy and the dispersion
    correction all exist -- none of which an implicit system exercises at all."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    root = tmp_path_factory.mktemp("hs-explicit")

    (root / "sys.config").write_text("solvent:\n  model: TIP3P\n", encoding="utf-8")
    built = subprocess.run(
        CLI + ["build-top", "-i", str(ALA), "-os", "built.xml", "-op", "built.pdb",
               "-log", "built.log", "--config", str(root / "sys.config")],
        cwd=root, capture_output=True, text=True, timeout=3600)
    assert built.returncode == 0, built.stdout + built.stderr

    import mdtraj

    frames = mdtraj.load(str(root / "built.pdb"))
    mdtraj.join([frames] * 4).save_dcd(str(root / "source.dcd"))

    # Shorter than the implicit run: a solvated box is two orders of magnitude more particles, and
    # what this fixture has to demonstrate is the identity through PME, not a long path.
    (root / "AIS.config").write_text(yaml.safe_dump({
        "protocol": "AIS", "solvent": "explicit",
        "ais": {"number_of_paths": 1, "switching_steps": 20,
                "observation_interval_steps": 5, "parameter_update_interval_steps": 5},
        "ais_source": {"trajectory": "../source.dcd"},
        "reporting": {"solute_printout": 10, "system_printout": 20,
                      "checkpoint_printout": 10}}), encoding="utf-8")
    generated = subprocess.run(CLI + ["build-md", "-odir", str(root / "project"),
                                      "--config", str(root / "AIS.config")],
                               capture_output=True, text=True, timeout=600)
    assert generated.returncode == 0, generated.stdout + generated.stderr

    done = _run_ais(root, root / "run")
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return root


def test_explicit_pme_hs_rows_match_an_independent_recomputation(explicit_run):
    """The identity through the reciprocal sum, the self-energy and the dispersion correction.

    An implicit system has none of those terms, so no implicit test reaches the part of the claim
    most likely to be wrong. Here the potentials are order 1e4 kJ/mol and the frame round trip is
    correspondingly larger in absolute terms -- which is why the tolerance scales with the energy
    rather than being a constant somebody chose while looking at a 22-particle system.
    """
    root, run = explicit_run, explicit_run / "run"
    rows = _hs_rows(run)
    assert rows, "no HS rows were written for the explicit run"

    worst = 0.0
    scale = 0.0
    for row in rows:
        tau = float(row["tau"])
        components, direct = recompute_at_frame(
            root, run, path_id=int(row["path_id"]),
            frame_index=int(row["coordinate_frame_index"]), tau=tau)
        allowed = frame_roundtrip_tolerance(float(row["potential_direct_kj_mol"]))
        for group in ("non_scaled", "sqrt_scaled", "lin_scaled"):
            recorded = float(row[f"potential_{group}_kj_mol"])
            scale = max(scale, abs(float(row["potential_direct_kj_mol"])))
            error = abs(recorded - getattr(components, group))
            worst = max(worst, error)
            assert error < allowed, (
                f"explicit frame {row['coordinate_frame_index']}: {group} recorded {recorded}, "
                f"recomputed {getattr(components, group)}")
        error = abs(float(row["potential_direct_kj_mol"]) - direct)
        worst = max(worst, error)
        assert error < allowed
    print(f"\nexplicit PME: max |recorded - recomputed| = {worst:.3e} kJ/mol on potentials up to "
          f"{scale:.3e} kJ/mol over {len(rows)} HS rows")


def test_explicit_pme_reconstruction_matches_the_direct_potential_in_the_file(explicit_run):
    """Both numbers are in the row; a reader checks the identity without recomputing anything."""
    from md_tools.ais.decomposition import reconstruction_tolerance
    from md_tools.build.record import read_record

    precision = (read_record(explicit_run / "run" / "AIS.log")["acceleration"].get(
        "cuda_precision") or "single")
    worst = 0.0
    for row in _hs_rows(explicit_run / "run"):
        tau = float(row["tau"])
        amplitude = 1.0 - tau
        reconstructed = (float(row["potential_non_scaled_kj_mol"])
                         + amplitude * float(row["potential_sqrt_scaled_kj_mol"])
                         + amplitude * amplitude * float(row["potential_lin_scaled_kj_mol"]))
        direct = float(row["potential_direct_kj_mol"])
        worst = max(worst, abs(reconstructed - direct))
        assert abs(reconstructed - direct) <= reconstruction_tolerance(
            direct, precision=precision), (row["switch_step"], reconstructed, direct)
    print(f"\nexplicit PME: max |reconstructed - direct| in the file = {worst:.3e} kJ/mol")
