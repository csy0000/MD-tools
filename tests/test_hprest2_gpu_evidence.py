"""GPU evidence for 0.5.3: a ladder exchanges, an extension JOINS its parent, restraints run.

Asked for by `docs/history/claudecode-instructions/20260913_gpu-tests-for-0.5.3.md`. The three fixes in
0.5.3 have CPU tests that prove a parser and a naming rule; none of them proves that a ladder still
runs on a GPU, that an extension continues rather than restarts, or that a restrained ladder is
sound. That is what this file is for, and every test here prints the numbers it measured rather
than only asserting -- a passed test is not the same as the number it produced.

WHAT IS AND IS NOT PROVABLE ABOUT A RESTRAINED LADDER

The instruction asks for a restrained and an unrestrained ladder, from the same seed and start, to
accept the SAME exchanges. That cannot hold, and expecting it would hide the real property: an
identical restraint cancels from `log alpha` EVALUATED AT THE SAME CONFIGURATIONS, but it also
changes the forces, so the two ladders' trajectories diverge from the first propagation and their
later decisions differ for an honest reason.

The exact statement is the energy identity, and it is tested two ways:

* `tests/test_ladder_torsion_restraints.py` computes `log alpha` with and without the bias at the
  same configurations and requires equality to 1e-6 kJ/mol;
* here, the same identity is checked on configurations taken from a REAL GPU ladder, so it is
  established for the states the sampler actually visits rather than for a synthetic pair.

Two GPUs, four rungs oversubscribed. Round-robin placement (`select_device_for_rank`) is retired:
`md_tools.openmm.placement` now puts two rungs on each device by measured throughput, and a shared
device is refused unless NVIDIA MPS is verified for every rank. On a machine without an MPS daemon
this file therefore refuses at the preflight; launch it from an MPS-enabled environment.

Nothing here names a platform. The cancellation check at the end therefore evaluates its energies
on CUDA in mixed precision like everything else in this file, and its tolerance says so: the EXACT
identity, to 1e-6 kJ/mol, is `tests/test_ladder_torsion_restraints.py`, which propagates nothing
and may legitimately use Reference. Splitting them that way is the suite's rule -- a file is
scientific GPU evidence or it is platform-exempt, never both.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import make_states_for

REPO = Path(__file__).resolve().parents[1]
ALA = REPO / "tests" / "data" / "ALA.pdb"
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.xdist_group("hprest2-gpu")]

RUNGS = 4
INTERVAL = 250
#: Long enough that the acceptance rates mean something. At 40 exchanges the ladder proposed 60
#: swaps, so a rate carried an uncertainty of about 6 percentage points and could not be compared
#: with anything; 200 exchanges is 300 proposals for a couple of minutes on two GPUs.
EXCHANGES = 200
EXTEND = 10


def report(name: str, numbers: dict) -> None:
    """Print what was measured, and keep it in a file the reporter can quote."""
    print(f"\n[{name}] " + json.dumps(numbers, indent=2, default=str), flush=True)
    out = os.environ.get("HPREST2_GPU_REPORT")
    if out:
        path = Path(out)
        existing = json.loads(path.read_text()) if path.is_file() else {}
        existing[name] = numbers
        path.write_text(json.dumps(existing, indent=2, default=str) + "\n", encoding="utf-8")


def _cli(work: Path, *args: str, timeout: int = 3600, launch=()):
    return subprocess.run([*launch, *CLI, *args], cwd=work, capture_output=True, text=True,
                          timeout=timeout)


def _digests(directory: Path) -> dict:
    """Every file under `directory`, by sha256. An extension must not modify its parent."""
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def _rem_log_rows(path: Path):
    """`rem.log` as `{exchange: {replica: neighbour}}`, in Amber's H-REMD layout.

    The exchange number is on a `# exchange N` COMMENT line; the data rows beneath it are
    `Rep# Neibr# Temp0 PotE(x_1) PotE(x_2) left_fe right_fe Success rate`. Reading three integers
    off a data row instead finds `300.00` in the third column and silently drops every row, which
    is what this parser did first: the log was well formed and the test saw nothing in it.
    """
    rows: dict[int, dict[int, int]] = {}
    exchange = None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.startswith("# exchange"):
            exchange = int(text.split()[-1])
            rows.setdefault(exchange, {})
            continue
        if text.startswith("#") or not text or exchange is None:
            continue
        fields = text.split()
        try:
            rows[exchange][int(fields[0])] = int(fields[1])
        except (ValueError, IndexError):
            continue
    return rows


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """ALA in implicit solvent, plus the CV and restraint definitions the third test needs."""
    if not ALA.is_file():
        pytest.skip("no ALA fixture")
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun on PATH; a ladder needs one rank per rung")
    work = tmp_path_factory.mktemp("hprest2-gpu")
    (work / "build").mkdir(exist_ok=True)
    (work / "sys.config").write_text("solvent:\n  model: GBn2\n", encoding="utf-8")
    done = _cli(work, "build-top", "-i", str(ALA), "-os", "build/built.xml", "-op", "build/built.pdb",
                "-log", "build/built.log", "--config", "sys.config")
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    shutil.copy2(REPO / "configs" / "md" / "cv.yaml", work / "cv.yaml")
    (work / "umbrella.yaml").write_text(
        "schema_version: 1\nrestraints:\n"
        "  - {cv: phi_ALA, form: harmonic, centre_deg: -60.0, force_constant: 500.0}\n",
        encoding="utf-8")
    return work


def _ladder_config(*, exchanges=EXCHANGES, restrained=False, cv=False) -> str:
    lines = [
        "protocol: REST2", "solvent: implicit",
        "dynamics: {timestep_fs: 2.0, temperature_K: 300.0, seed: 20260913}",
        "stages: {minimization_iterations: 0, restrained_nvt_steps: 500, "
        "restrained_npt_steps: 0, unrestrained_npt_steps: 0, production_steps: 0}",
        "reporting: {crd_printout_solute: %d, info_printout: %d, checkpoint_printout: %d}"
        % (INTERVAL, INTERVAL, INTERVAL),
    ]
    if cv or restrained:
        lines.append("collective_variables: {file: cv.yaml, interval_steps: %d}" % INTERVAL)
    if restrained:
        lines.append("umbrella: {file: umbrella.yaml}")
    lines.append(
        "rest2: {number_of_replicas: %d, tau_max: 0.5, exchange_interval_steps: %d, "
        "number_of_exchanges: %d, state_trajectory: true, rem_log: true, "
        "neighbour_acceptance_report: true}" % (RUNGS, INTERVAL, exchanges))
    return "\n".join(lines) + "\n"


def _build_and_equilibrate(built: Path, name: str, config: str) -> Path:
    """One SYSTEM ROOT PER VARIANT, sharing the one built System.

    `input/` is shared by every run on a system and the sharing is enforced: `input/eq_1.in` is
    refused if a second configuration resolves it differently. The `restrained` variant adds
    `collective_variables` and `umbrella` to the very same equilibration stage as `baseline`, so
    the two resolve that shared input differently and the second generation was refused with

        build-md: input/eq_1.in already exists and is not what this configuration resolves to.

    They are different experiments over one system, not repeats of one, so each gets its own root
    -- which is what the refusal advises. The definitions are copied in because the configuration
    names them by bare name, resolved beside the configuration file.
    """
    root = built / f"system-{name}"
    shutil.copytree(built / "build", root / "build")
    for helper in ("cv.yaml", "umbrella.yaml"):
        shutil.copy2(built / helper, root / helper)

    (root / f"{name}.config").write_text(config, encoding="utf-8")
    # A REST2 ladder integrates SAVED scaled states (0.5.4): `build-md` refuses to generate one
    # until `build-top --rest2-scaler` has written build/REST2/ beside the System.
    make_states_for(root, root / f"{name}.config")
    done = _cli(root, "build-md", "-odir", f"./{name}-run1", "--config", f"{name}.config")
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = root / f"{name}-run1"
    # `-odir eq`, AND NO `-r`/`-chk`/`-o`/`-log`.
    #
    # The equilibration stage belongs to the run's `eq/`, and md-run names all four artefacts
    # inside `-odir` under the stage's FILING KEY -- so this stage writes `eq/eq_1.xml`. A value
    # given explicitly is taken verbatim against the WORKING directory instead, which is how
    # `-odir .` with `-r eq.xml` put the restart at the run root while the run's own
    # `resolved.config` there described a REST2 ladder: resolving the method-neutral
    # `../input/eq_1.in` against it was refused with `protocol: was 'REST2', now 'cMD'`.
    done = _cli(run, "md-run", "-i", "../input/eq_1.in", "-p", "../build/built.pdb",
                "-s", "../build/built.xml", "-odir", "eq")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    # THE REST OF THE GENERATED CHAIN, zero-length here. Every line of the group file `build-md`
    # wrote continues from `-c eq/eq_3.xml`, and a ladder reads its inputs only from that file
    # (0.5.4), so the ladder no longer takes `-c eq/eq_1.xml` on its command line. Running the two
    # zero-step stages is what run.sh types; with no steps they carry eq_1's configuration forward.
    for key, parent in (("eq_2", "eq/eq_1.xml"), ("eq_3", "eq/eq_2.xml")):
        done = _cli(run, "md-run", "-i", f"../input/{key}.in", "-p", "../build/built.pdb",
                    "-s", "../build/built.xml", "-c", parent, "-odir", "eq")
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return run


#: The helpers an extension's own directory needs. `_protocol.py` and `solute.yaml` are written by
#: rank 0 at run time, into `-odir`. THE GROUP FILE IS COPIED TOO (0.5.4): the runtime writes none,
#: a ladder reads -s only from one, and its `-i _protocol.py` resolves beside the group file -- so
#: the group file must sit in the directory the extension writes into, or the launch is refused.
#: Its `../build/` paths still resolve, the extension being a sibling of the run. The `-c
#: eq/eq_3.xml` every line names is copied with it, since the line names it; an extension does not
#: read it, which `test_supplying_coordinates_to_an_extension_is_inert` is about.
#:
#: NO `.in`. The generated `REST2.py` reads `resolved.config` beside itself and nothing else --
#: `run_generated_remd(__file__, protocol="REST2")` is its whole body -- and `_generated` passes no
#: `-i`. The Amber-like inputs belong to the SYSTEM and live in `<system>/input/`, so a run
#: directory holds none to copy: this named `REST2.in` and every extension test died with
#: `FileNotFoundError: .../baseline-run1/REST2.in`. An extension directory is a sibling of the run
#: inside the same system root, so `../input/` still resolves from it if anything ever needs it.
EXTENSION_FILES = ("REST2.py", "resolved.config", "remd_groupfile.1", "eq/eq_3.xml")


def _generated(work: Path, *extra: str, timeout: int = 3600):
    """The GENERATED entry point under mpirun: what a `build-md` tree gives a user.

    `md-openmm md-run` defines neither `--extend` nor `--extend-from` -- its parser stops at
    `--resume`/`--overwrite` -- and `REST2.py` forwards both to the executor. So this is the only
    way to launch an extension of a generated ladder, and using `md-run` here would fail in
    argparse before any of the behaviour under test ran.

    NO -s: a ladder reads its saved states only from a group file (0.5.4), the copy of the
    parent's in the extension's own directory (see EXTENSION_FILES).
    """
    return subprocess.run(["mpirun", "-n", str(RUNGS), sys.executable, "REST2.py",
                           "-p", "../build/built.pdb", "--groupfile", "remd_groupfile.1",
                           "-ng", str(RUNGS), *extra],
                          cwd=str(work), capture_output=True, text=True, timeout=timeout)


def _run_ladder(run: Path, *extra: str, timeout: int = 3600):
    return _cli(run, "md-run", "-ng", str(RUNGS), "-i", "../input/REST2.in", "-p", "../build/built.pdb",
                "--groupfile", "remd_groupfile.1", "-x", "REST2.nc", "-r", "restart.json", "-o", "REST2.out",
                "-log", "REST2.log", *extra, launch=["mpirun", "-n", str(RUNGS)], timeout=timeout)


# ---------------------------------------------------------------------------------------------
# 1. A ladder still runs and still exchanges
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def baseline(built):
    run = _build_and_equilibrate(built, "baseline", _ladder_config())
    done = _run_ladder(run)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return run


def test_a_ladder_runs_and_every_exchange_is_a_permutation(baseline):
    manifest = json.loads((baseline / "restart.json").read_text(encoding="utf-8"))
    rows = _rem_log_rows(baseline / "rem.log")
    assert rows, "rem.log has no exchange rows"
    for exchange, pairing in rows.items():
        assert sorted(pairing) == list(range(1, RUNGS + 1)), \
            f"exchange {exchange} has rows {sorted(pairing)}, not one per rung"
        # Each row names its partner, so the pairing must be an involution: if 1 names 2, 2 names
        # 1. A log in which a replica's partner does not name it back describes swaps that cannot
        # all have happened.
        for replica, neighbour in pairing.items():
            assert pairing[neighbour] == replica, \
                f"exchange {exchange}: replica {replica} names {neighbour}, which names " \
                f"{pairing[neighbour]}"
    final = list(manifest["final_state_to_walker"])
    assert sorted(final) == list(range(RUNGS)), "the final map is not a permutation"
    statistics = manifest.get("lifetime_statistics") or {}
    report("1-baseline", {
        "run_status": manifest["run_status"],
        "exchanges_committed": manifest["exchanges_committed"],
        "exchanges_asked_for": EXCHANGES,
        "steps_completed": manifest["steps_completed"],
        "final_state_to_walker": final,
        "total_proposed": statistics.get("total_proposed"),
        "total_accepted": statistics.get("total_accepted"),
        "overall_acceptance": statistics.get("overall_acceptance"),
        "by_state_pair": statistics.get("by_state_pair"),
        "mapping_integrity": manifest.get("mapping_integrity"),
        "rem_log_exchanges": len(rows),
    })
    assert (manifest.get("mapping_integrity") or {}).get("every_row_is_a_permutation") is True
    assert manifest["run_status"] == "completed"
    assert manifest["exchanges_committed"] == EXCHANGES


# ---------------------------------------------------------------------------------------------
# 2. An extension JOINS its parent
# ---------------------------------------------------------------------------------------------
def test_an_extension_continues_its_parent_without_coordinates(baseline, tmp_path):
    """`--extend-from` with no `-c`: the fix. What it produces must be a continuation."""
    parent = json.loads((baseline / "restart.json").read_text(encoding="utf-8"))
    before = _digests(baseline)
    extension = baseline.parent / "extension"
    extension.mkdir()
    for helper in EXTENSION_FILES:
        (extension / helper).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline / helper, extension / helper)

    # Through the GENERATED entry point, which is what a build-md tree gives a user: `md-run`
    # defines neither --extend nor --extend-from, and `REST2.py` forwards both to the executor.
    done = _generated(extension, "--extend", str(EXTEND), "--extend-from", str(baseline))
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]

    child = json.loads((extension / "restart.json").read_text(encoding="utf-8"))
    after = _digests(baseline)

    # THE PART A UNIT TEST CANNOT REACH: does the extension's trajectory JOIN its parent's, or did
    # it re-equilibrate? Every check above passes either way. The scale for "joined" is the
    # parent's own frame-to-frame movement: consecutive frames are one exchange interval apart, and
    # the child's first frame is exactly one such interval after the parent's last. A restart from
    # re-thermalised coordinates would be a far bigger jump -- compared here against the parent's
    # own first-to-last spread, which is what "a different configuration" looks like.
    import numpy as np
    from netCDF4 import Dataset

    def frames(path):
        with Dataset(str(path)) as dataset:
            return np.array(dataset.variables["coordinates"][:], dtype=float)

    def occupancy(run):
        with Dataset(str(run / "REST2.nc")) as dataset:
            return np.array(dataset.variables["state_to_walker"][:], dtype=int)

    def rms(a, b):
        return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum(axis=-1).mean()))

    # MATCHED BY WALKER, not by state. A per-state file changes occupant at every accepted
    # exchange, so the parent's last frame of state s and the child's first frame of state s are
    # usually different molecules and comparing them measures the exchange, not the join. Frames
    # are written with the mapping in force, so the mapping says which state each walker was in:
    # the parent's final row, and the child's first.
    parent_map, child_map = occupancy(baseline)[-1], occupancy(extension)[0]
    continuity = {}
    for walker in range(RUNGS):
        from_state = int(np.where(parent_map == walker)[0][0])
        into_state = int(np.where(child_map == walker)[0][0])
        parent_frames = frames(baseline / f"solute_state{from_state}_prod1.nc")
        child_frames = frames(extension / f"solute_state{into_state}_prod1.nc")
        continuity[walker] = {
            "parent_state": from_state,
            "child_state": into_state,
            "walker_matched_join_angstrom": rms(parent_frames[-1], child_frames[0]),
            # The same comparison done by state, which is what the earlier version measured.
            "same_state_join_angstrom": rms(
                parent_frames[-1], frames(extension / f"solute_state{from_state}_prod1.nc")[0]),
            "one_interval_within_parent_angstrom": rms(parent_frames[-2], parent_frames[-1]),
            "parent_first_to_last_angstrom": rms(parent_frames[0], parent_frames[-1]),
        }
    report("2-extension", {
        "parent_steps_completed": parent["steps_completed"],
        "child_resumed_from_step": child.get("resumed_from_step"),
        "child_steps_completed": child["steps_completed"],
        "expected_child_steps": parent["steps_completed"] + EXTEND * INTERVAL,
        "parent_final_state_to_walker": list(parent["final_state_to_walker"]),
        "child_final_state_to_walker": list(child["final_state_to_walker"]),
        "parent_unchanged": before == after,
        "parent_files_changed": sorted(k for k in before if before[k] != after.get(k)),
        "trajectory_continuity": continuity,
    })
    for walker, measured in continuity.items():
        # One exchange interval of dynamics for the SAME molecule must move it less than the
        # parent's own first-to-last spread, which is what "somewhere else entirely" looks like.
        assert (measured["walker_matched_join_angstrom"]
                < measured["parent_first_to_last_angstrom"]), (
            f"walker {walker}: it is {measured['walker_matched_join_angstrom']:.3f} A from where "
            f"the parent left it, against {measured['parent_first_to_last_angstrom']:.3f} A "
            f"across the parent's whole run -- that is a restart, not a continuation")
    assert child["resumed_from_step"] == parent["steps_completed"], "not a continuation"
    assert child["steps_completed"] == parent["steps_completed"] + EXTEND * INTERVAL
    assert sorted(child["final_state_to_walker"]) == list(range(RUNGS))
    assert before == after, "the extension modified its parent"


def test_supplying_coordinates_to_an_extension_is_inert(baseline, tmp_path):
    """`-c` on an extension is accepted and ignored; if it were read, the runs would differ."""
    parent = json.loads((baseline / "restart.json").read_text(encoding="utf-8"))
    results = {}
    for name, extra in (("without_c", ()), ("with_c", ("-c", str(baseline / "eq" / "eq_1.xml")))):
        out = baseline.parent / f"inert-{name}"
        out.mkdir()
        for helper in EXTENSION_FILES:
            (out / helper).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline / helper, out / helper)
        done = _generated(out, "--extend", "2", "--extend-from", str(baseline), *extra)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        manifest = json.loads((out / "restart.json").read_text(encoding="utf-8"))
        results[name] = {"resumed_from_step": manifest.get("resumed_from_step"),
                         "steps_completed": manifest["steps_completed"],
                         "final_state_to_walker": list(manifest["final_state_to_walker"]),
                         "rem_log": _rem_log_rows(out / "rem.log")}
    report("2b-inert-c", {
        "parent_steps_completed": parent["steps_completed"],
        "without_c": {k: v for k, v in results["without_c"].items() if k != "rem_log"},
        "with_c": {k: v for k, v in results["with_c"].items() if k != "rem_log"},
        "same_exchange_decisions": results["without_c"]["rem_log"] == results["with_c"]["rem_log"],
    })
    assert results["without_c"]["rem_log"] == results["with_c"]["rem_log"], \
        "passing -c changed the run, so coordinates are being read on a path that claims not to"


# ---------------------------------------------------------------------------------------------
# 3. A restrained ladder
# ---------------------------------------------------------------------------------------------
def test_a_restrained_ladder_runs_and_its_bias_cancels_from_the_exchange(built):
    """The restraint holds the torsion, and cancels from `log alpha` on the run's OWN states."""
    import numpy as np
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile

    from md_tools.remd.protocol import build_rung_systems

    run = _build_and_equilibrate(built, "restrained", _ladder_config(exchanges=20,
                                                                    restrained=True))
    done = _run_ladder(run)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    manifest = json.loads((run / "restart.json").read_text(encoding="utf-8"))
    restraints = ((manifest.get("scientific_identity") or {}).get("torsion_restraints") or {})

    # The restrained torsion, as the run itself reported it.
    import csv

    values, start_values = [], []
    for state in range(RUNGS):
        path = run / f"cv_state{state}.csv"
        if not path.is_file():
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if "phi_ALA" not in row:
                    continue
                # The step-0 observation, marked `exchange_attempt = -1`, is taken BEFORE a single
                # step is propagated: it is the shared, unrestrained starting configuration, one
                # per state, and it is where phi still sits at about -160 degrees. Judging the
                # restraint by it measures the structure the ladder began from, not the hold.
                if int(row.get("exchange_attempt", 0)) < 0:
                    start_values.append(float(row["phi_ALA"]))
                else:
                    values.append(float(row["phi_ALA"]))
    def deviations(series):
        return [abs((v - (-60.0) + 180.0) % 360.0 - 180.0) for v in series]

    deviation = deviations(values)
    started_at = deviations(start_values)

    # The identity, on configurations this ladder actually visited.
    system = XmlSerializer.deserialize((built / "build" / "built.xml").read_text(encoding="utf-8"))
    pdb = PDBFile(str(built / "build" / "built.pdb"))
    solute = list(range(system.getNumParticles()))
    taus = [float(state["tau"]) for state in manifest["states"]][:2]
    records = restraints.get("restraints") or []
    plain, _a = build_rung_systems(system, solute, taus)
    biased, _b = build_rung_systems(system, solute, taus, restraints=records)

    from openmm import Context, VerletIntegrator

    def energy(system_, positions):
        # No platform named: this file's Contexts are the machine's default, which is CUDA.
        context = Context(system_, VerletIntegrator(0.001 * unit.picoseconds))
        try:
            context.setPositions(positions)
            return float(context.getState(getEnergy=True).getPotentialEnergy()
                         .value_in_unit(unit.kilojoule_per_mole))
        finally:
            del context

    x_i = np.array(pdb.positions.value_in_unit(unit.nanometer))
    x_j = x_i + 0.02 * np.random.default_rng(7).standard_normal(x_i.shape)
    frames = [x_i * unit.nanometer, x_j * unit.nanometer]

    def log_alpha(systems):
        u = [[energy(systems[rung], frames[which]) for which in (0, 1)] for rung in (0, 1)]
        return (u[0][1] + u[1][0]) - (u[0][0] + u[1][1])

    difference = log_alpha(biased) - log_alpha(plain)
    report("3-restrained", {
        "run_status": manifest["run_status"],
        "exchanges_committed": manifest["exchanges_committed"],
        "restraint_record": restraints,
        "phi_propagated_samples": len(values),
        "phi_start_samples_excluded": len(start_values),
        "phi_start_deviation_deg": max(started_at) if started_at else None,
        "phi_mean_abs_deviation_from_centre_deg": (sum(deviation) / len(deviation)
                                                   if deviation else None),
        "phi_max_abs_deviation_deg": max(deviation) if deviation else None,
        "log_alpha_difference_kj_mol": difference,
        "tolerance_kj_mol": 0.05,
        "note": ("a restrained and an unrestrained ladder do NOT accept the same exchanges: the "
                 "bias changes the forces, so their trajectories diverge. What cancels is the "
                 "criterion at the SAME configurations, which is what is measured here. The "
                 "tolerance is mixed-precision CUDA noise on ~1e4 kJ/mol energies; the exact "
                 "identity, to 1e-6, is tests/test_ladder_torsion_restraints.py."),
    })
    assert manifest["run_status"] == "completed"
    assert restraints.get("scaled_by_tau") is False
    assert abs(difference) < 0.05, "the restraint did not cancel from the exchange criterion"
    if deviation:
        assert max(deviation) < 90.0, "the restraint did not hold the torsion"
        assert sum(deviation) / len(deviation) < 30.0, \
            "the restrained torsion sits far from its centre on average"
