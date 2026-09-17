"""Regressions for the six corrections review found in the `md-run` surface.

Written BEFORE the implementation, so each names a defect that exists at the baseline
`da24bde`. They supersede earlier tests that encoded the mappings being corrected here, and each
of those is replaced rather than deleted -- the docstring in its file says what superseded it.

1. **Amber short flags.** `-x` was the serialised System and the trajectory had no short flag.
   Amber spells `-x` mdcrd and nothing else; `-s` is the MD-tools extension, because a PDB does
   not carry parameters.
2. **`-o` and `-log`.** They were aliased to one artefact, on the argument that this package
   writes one readable file. That conflated two audiences: a person watching a run, and the
   provenance record a machine reads.
3. **The platform is a machine property**, not a scientific one. It lived in `dynamics.platform`
   in every protocol config, where a workflow shared between machines carried one machine's
   hardware.
4. **AIS accepted reporting settings that did nothing.** `info_printout` and
   `checkpoint_printout` were validated and then never read, and `crd_printout_solute` was forced
   equal to the observation interval. A strict input language must not accept a consequential
   setting it ignores.
5. **Trajectory formats were taken on faith.** Nothing verified that a `.dcd` was a DCD or that a
   `.nc` was NetCDF, and AIS trusted the suffix of its source.
6. **MPI failed open.** `barrier` returned silently when `mpi4py` was missing, so a launch of N
   ranks with no working MPI ran N uncoordinated simulations over one set of output paths.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "md_tools.cli.md_openmm"]
ALA = REPO / "tests" / "data" / "ALA.pdb"


def _written(text: str, name: str = "run.in") -> Path:
    path = Path(tempfile.mkdtemp()) / name
    path.write_text(text, encoding="utf-8")
    return path


MINIMAL_CMD = "&cntrl\n  protocol = cMD,\n  stage    = min,\n/\n"


# --- 1. Amber short-flag semantics ------------------------------------------------------------

def test_dash_s_is_the_serialised_system_and_dash_x_is_the_trajectory():
    """Supersedes the mapping in which `-x` named `built.xml`.

    Amber's `-x` is mdcrd and has meant that for decades. Repurposing it for the System made every
    example in this project read wrong to the audience the surface exists for.
    """
    from md_tools.run.main import md_run_parser

    args = md_run_parser().parse_args(
        ["-i", "min.in", "-p", "built.pdb", "-s", "built.xml", "-x", "min.dcd"])
    assert args.system == "built.xml"
    assert args.trajectory == "min.dcd"


def test_the_system_is_mandatory_and_the_trajectory_is_optional():
    """`-s` or `-groupfile`, exactly one. The trajectory stays optional.

    MIGRATED. `-s` used to be `required=True` in the parser, and this asserted that argparse
    refused its absence. That made a grouped REST2/rREST2 launch impossible through `md-run`: a
    ladder's rungs are scaled and serialised at build time, so each line of the group file names
    its OWN pre-scaled System and there is no single `-s` for the launch to carry. argparse
    refused before the runtime's grouped exemption could be reached, and `run.sh` -- the
    documented way to run a ladder -- died on every rank.

    So the rule is now "exactly one of the two", enforced by name in the flag-role checks rather
    than by argparse. The mandatory half is still asserted, and more precisely than before: a
    launch naming NEITHER is refused, and one naming BOTH is refused too, because they are two
    answers to one question -- which System each replica integrates.
    """
    from md_tools.run.main import _check_file_roles, md_run_parser

    args = md_run_parser().parse_args(["-i", "min.in", "-p", "built.pdb", "-s", "built.xml"])
    assert args.trajectory is None, "a stage supplies its own default trajectory name"

    # A ladder: no `-s`, and the parser must let it through to the rule below.
    grouped = md_run_parser().parse_args(
        ["-i", "REST2.in", "-p", "built.pdb", "-groupfile", "remd.group", "-ng", "3"])
    assert grouped.system is None and grouped.groupfile == "remd.group"
    _check_file_roles(grouped)          # accepted: the group file says what to integrate

    # Neither: still refused, now by name.
    with pytest.raises(SystemExit) as neither:
        _check_file_roles(md_run_parser().parse_args(["-i", "min.in", "-p", "built.pdb"]))
    assert "-groupfile" in str(neither.value) and "-s" in str(neither.value), str(neither.value)

    # Both: refused, because one `-s` beside a group file claims one Hamiltonian for every rung.
    with pytest.raises(SystemExit) as both:
        _check_file_roles(md_run_parser().parse_args(
            ["-i", "REST2.in", "-p", "built.pdb", "-s", "built.xml",
             "-groupfile", "remd.group"]))
    assert "both" in str(both.value).lower(), str(both.value)


def test_a_system_passed_as_the_trajectory_is_refused_by_name():
    """`-x built.xml` is the exact mistake the old mapping trained people to make."""
    done = subprocess.run(
        CLI + ["md-run", "-i", "min.in", "-p", "built.pdb", "-s", "built.xml", "-x", "built.xml"],
        capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "-x" in done.stderr and "-s" in done.stderr, done.stderr
    assert "trajectory" in done.stderr.lower(), done.stderr


def test_a_trajectory_passed_as_the_system_is_refused_by_name():
    done = subprocess.run(
        CLI + ["md-run", "-i", "min.in", "-p", "built.pdb", "-s", "min.dcd"],
        capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "-s" in done.stderr, done.stderr


def test_the_retired_platform_option_is_rejected():
    """The platform moved to `machine.openmm`; a per-run `--platform` would be a second authority."""
    done = subprocess.run(
        CLI + ["md-run", "-i", "min.in", "-p", "built.pdb", "-s", "built.xml",
               "--platform", "CUDA"],
        capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "platform" in done.stderr.lower(), done.stderr


def test_an_abbreviated_flag_is_not_silently_reinterpreted():
    """argparse abbreviates unique prefixes. `--traj` must not become `--trajectory` by accident.

    A misspelled flag that argparse resolves is worse than one that fails: it runs, with a setting
    the writer did not mean to give.
    """
    from md_tools.run.main import md_run_parser

    with pytest.raises(SystemExit):
        md_run_parser().parse_args(
            ["-i", "min.in", "-p", "built.pdb", "-s", "built.xml", "--traj", "min.dcd"])


# --- 2. `-o` and `-log` are two artefacts -----------------------------------------------------

def test_output_and_log_are_separate_files_with_separate_defaults():
    """Supersedes `_reconcile_output_and_log`, which made `-o` a second spelling of `-log`.

    They serve different readers. `.out` is what a person opens while a run is going; `.log` is
    the provenance record a machine reads. Collapsing them forced a human to read a machine
    record to see whether a run was making progress.
    """
    from md_tools.run.main import md_run_parser

    args = md_run_parser().parse_args(
        ["-i", "min.in", "-p", "built.pdb", "-s", "built.xml",
         "-o", "min.out", "-log", "min.log"])
    assert args.output == "min.out" and args.log == "min.log"
    assert not hasattr(args, "_output_and_log_differ"), "the reconciliation must be gone"


def test_the_same_path_for_output_and_log_is_refused():
    """Different roles, so one file cannot be both. Only an actual collision is refused."""
    done = subprocess.run(
        CLI + ["md-run", "-i", "min.in", "-p", "built.pdb", "-s", "built.xml",
               "-o", "same.txt", "-log", "same.txt"],
        capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "same.txt" in done.stderr, done.stderr


# --- 3. the platform is a machine property ----------------------------------------------------

def test_machine_openmm_defaults_exist_and_are_cuda_mixed_local_rank():
    from md_tools.registry.userconfig import resolve_machine_openmm

    resolved = resolve_machine_openmm({})
    assert resolved["platform"] == "CUDA"
    assert resolved["precision"] == "mixed"
    assert resolved["device_policy"] == "local_rank"
    assert resolved["origin"] == "built-in default"


def test_a_machine_configuration_can_choose_cpu_and_it_is_recorded_as_such():
    """A machine-wide CPU default is legitimate; it must not look like an unnoticed fallback."""
    from md_tools.registry.userconfig import resolve_machine_openmm

    resolved = resolve_machine_openmm(
        {"machine": {"openmm": {"platform": "CPU", "precision": "mixed"}}})
    assert resolved["platform"] == "CPU"
    assert resolved["origin"] == "machine.openmm"


def test_a_user_configuration_without_the_openmm_block_is_still_valid():
    from md_tools.registry.userconfig import resolve_machine_openmm

    resolved = resolve_machine_openmm({"machine": {"md_data": "/tmp"}})
    assert resolved["platform"] == "CUDA" and resolved["origin"] == "built-in default"


def test_the_shipped_user_configuration_example_documents_the_openmm_block():
    text = (REPO / "configs" / "machine" / "user.config.example").read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    assert document["machine"]["openmm"] == {"platform": "CUDA", "precision": "mixed",
                                             "device_policy": "local_rank"}, document["machine"]


def test_init_writes_the_machine_openmm_defaults(tmp_path):
    from md_tools.registry.userconfig import init_user_config

    path = init_user_config(explicit=tmp_path / "user.config", noninteractive=True,
                            name="A Person", person_id="a-person", md_data=str(tmp_path))
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["machine"]["openmm"]["platform"] == "CUDA"


def test_the_retired_dynamics_platform_key_is_rejected_with_a_migration_message():
    from md_tools.build.md import resolve_md_config
    from md_tools.build.strict import ConfigError

    path = _written("protocol: cMD\ndynamics:\n  platform: CUDA\n", "md.config")
    with pytest.raises(ConfigError) as refusal:
        resolve_md_config(path)
    assert "machine.openmm.platform" in str(refusal.value), refusal.value


def test_no_shipped_protocol_configuration_states_a_platform():
    for name in ("cMD", "REST2", "AIS"):
        text = (REPO / "configs" / "md" / f"{name}.config").read_text(encoding="utf-8")
        document = yaml.safe_load(text)
        assert "platform" not in (document.get("dynamics") or {}), name


# --- 4. every accepted AIS reporting option does something ------------------------------------

def _ais_document(**reporting):
    return {"protocol": "AIS", "solvent": "explicit",
            "ais": {"number_of_paths": 4, "switching_steps": 250,
                    "observation_interval_steps": 10},
            "ais_source": {"trajectory": "../source.dcd"},
            "reporting": {"crd_printout_solute": 10, "info_printout": 50,
                          "checkpoint_printout": 50, **reporting}}


def test_the_four_ais_cadences_may_all_differ():
    """Supersedes the rule that forced `crd_printout_solute == observation_interval_steps`.

    They are four different questions -- how often work is measured, how often a configuration is
    written, how often the state table is sampled, how often the run becomes resumable -- and
    tying two of them together answered one of the questions wrongly.
    """
    from md_tools.build.md import resolve_md_config

    path = _written(yaml.safe_dump(_ais_document(crd_printout_solute=25, info_printout=50,
                                                 checkpoint_printout=125)), "AIS.config")
    resolved = resolve_md_config(path)
    assert resolved["reporting"]["crd_printout_solute"] == 25
    assert resolved["ais"]["observation_interval_steps"] == 10


@pytest.mark.parametrize("stream", ["crd_printout_solute", "info_printout", "checkpoint_printout"])
def test_an_ais_cadence_that_does_not_divide_the_path_is_refused(stream):
    from md_tools.build.md import resolve_md_config
    from md_tools.build.strict import ConfigError

    path = _written(yaml.safe_dump(_ais_document(**{stream: 100})), "AIS.config")
    with pytest.raises(ConfigError) as refusal:
        resolve_md_config(path)
    assert "250" in str(refusal.value) and "100" in str(refusal.value)


def test_the_ais_runtime_is_given_the_reporting_block():
    """It was validated and then dropped: `ais_main` never saw `reporting` at all."""
    import inspect

    from md_tools.ais.run import run_generated_ais

    source = inspect.getsource(run_generated_ais)
    assert '"reporting"' in source, "the resolved reporting block never reaches the AIS runtime"


def test_the_ais_schedule_carries_the_four_independent_cadences():
    from md_tools.ais.schedule import switching_schedule

    schedule = switching_schedule(
        tau_start=0.5, tau_end=0.0, switching_steps=250,
        parameter_update_interval_steps=1, observation_interval_steps=10,
        timestep_fs=2.0, trajectory_interval_steps=25, state_interval_steps=50,
        checkpoint_interval_steps=125)
    assert schedule["number_of_observations"] == 26
    assert schedule["number_of_frames"] == 11               # steps 0, 25 … 250
    assert schedule["number_of_state_rows"] == 6            # steps 0, 50 … 250
    assert schedule["checkpoint_steps"] == [125, 250]


# --- 5. genuine formats -----------------------------------------------------------------------

def test_a_stage_accepts_dcd_and_netcdf_and_refuses_anything_else():
    """A stage now writes AMBER NetCDF, so `.nc` is honest -- and the guard still has a job.

    THIS TEST ASSERTED THE OPPOSITE, and the reason it did has been overtaken rather than
    disproved. It read:

        The ordinary writer is OpenMM's DCDReporter. Renaming a DCD does not make it NetCDF.

    True while the stage used OpenMM's reporter, which has no NetCDF writer. The stage now writes
    through mdtraj's `NetCDFReporter`, which produces genuine AMBER NetCDF -- and must, because
    that is the only format here that carries an atom SUBSET, which is what a solute-only stream
    is.

    What the guard still refuses is unchanged and is what this now checks: a name claiming a
    format nothing writes. Every reader downstream opens a trajectory by extension, so a file
    whose name and contents disagree fails in the reader and gets blamed on the reader.
    """
    from md_tools.md.stage import check_trajectory_suffix

    check_trajectory_suffix(Path("solute_prod1.nc"))         # accepted: what a stage writes
    check_trajectory_suffix(Path("cMD.dcd"))                 # accepted: still supported
    for claimed in ("cMD.xtc", "cMD.trr", "cMD.pdb", "cMD"):
        with pytest.raises(SystemExit) as refusal:
            check_trajectory_suffix(Path(claimed))
        assert "DCD or AMBER NetCDF" in str(refusal.value), refusal.value


def test_the_trajectory_format_of_a_file_is_read_from_its_contents():
    """Suffix and content can disagree, and only the content decides what a reader will find."""
    from md_tools.openmm.trajectory import detect_trajectory_format

    dcd = Path(tempfile.mkdtemp()) / "looks_like.nc"
    dcd.write_bytes(b"\x54\x00\x00\x00CORD" + b"\x00" * 200)
    assert detect_trajectory_format(dcd) == "dcd"

    netcdf = Path(tempfile.mkdtemp()) / "looks_like.dcd"
    netcdf.write_bytes(b"CDF\x01" + b"\x00" * 200)
    assert detect_trajectory_format(netcdf) == "netcdf"


def test_ais_refuses_a_source_whose_suffix_and_contents_disagree():
    from md_tools.openmm.trajectory import check_trajectory_declaration

    mislabelled = Path(tempfile.mkdtemp()) / "source.nc"
    mislabelled.write_bytes(b"\x54\x00\x00\x00CORD" + b"\x00" * 200)
    with pytest.raises(SystemExit) as refusal:
        check_trajectory_declaration(mislabelled)
    message = str(refusal.value)
    assert ".nc" in message and "dcd" in message.lower(), message


# --- 6. MPI fails closed ----------------------------------------------------------------------

def test_a_multi_rank_launch_without_mpi4py_is_a_fatal_preflight_error():
    """Supersedes the permissive barrier, which returned silently when mpi4py was missing.

    Eight ranks with no coordination are not a slow run: they are eight simulations writing over
    one set of output paths, each believing it is the whole ladder.
    """
    from md_tools.remd.mpi import require_mpi

    with pytest.raises(SystemExit) as refusal:
        require_mpi(size=4, importer=lambda: (_ for _ in ()).throw(ImportError("no mpi4py")))
    message = str(refusal.value)
    assert "mpi4py" in message and "4" in message, message


def test_a_serial_launch_needs_no_mpi4py_at_all():
    from md_tools.remd.mpi import require_mpi

    assert require_mpi(size=1, importer=lambda: (_ for _ in ()).throw(ImportError())) is None


def test_a_launcher_size_that_disagrees_with_the_communicator_is_refused():
    from md_tools.remd.mpi import check_launch_consistency

    with pytest.raises(SystemExit) as refusal:
        check_launch_consistency(launcher_rank=0, launcher_size=8,
                                 comm_rank=0, comm_size=4, number_of_groups=8, replicas=8)
    assert "8" in str(refusal.value) and "4" in str(refusal.value)


def test_a_group_count_that_disagrees_with_the_communicator_is_refused():
    from md_tools.remd.mpi import check_launch_consistency

    with pytest.raises(SystemExit):
        check_launch_consistency(launcher_rank=0, launcher_size=4,
                                 comm_rank=0, comm_size=4, number_of_groups=8, replicas=8)


def test_a_replica_count_that_disagrees_with_the_group_count_is_refused():
    from md_tools.remd.mpi import check_launch_consistency

    with pytest.raises(SystemExit):
        check_launch_consistency(launcher_rank=0, launcher_size=4,
                                 comm_rank=0, comm_size=4, number_of_groups=4, replicas=8)


def test_a_consistent_launch_is_accepted():
    from md_tools.remd.mpi import check_launch_consistency

    assert check_launch_consistency(launcher_rank=1, launcher_size=4, comm_rank=1, comm_size=4,
                                    number_of_groups=4, replicas=4) is None


def test_a_multi_rank_preflight_failure_leaves_no_authoritative_output(tmp_path):
    """No `resolved.config`, no `.out`, no `.log`, no trajectory: nothing to mistake for a run."""
    environment = dict(os.environ, OMPI_COMM_WORLD_RANK="0", OMPI_COMM_WORLD_SIZE="4",
                       MD_TOOLS_FORCE_NO_MPI4PY="1")
    (tmp_path / "REST2.in").write_text(
        "&cntrl\n  protocol = REST2,\n/\n&remd\n  number_of_replicas = 4,\n/\n", encoding="utf-8")
    # Real files, so the MPI check is genuinely REACHED. Preflight validates the inputs before the
    # launch -- an unreadable topology is wrong on every machine, and its message should not be
    # preceded by one about MPI -- so a test that leaves them missing refuses for a different
    # reason than the one it is named for.
    (tmp_path / "built.pdb").write_text("END\n", encoding="utf-8")
    (tmp_path / "built.xml").write_text("<System/>\n", encoding="utf-8")
    # A ladder reads -s only from its group file (0.5.4).
    (tmp_path / "state.xml").write_text("<State/>\n", encoding="utf-8")
    (tmp_path / "group").write_text("".join(
        f"-i p.py -p built.pdb -s built.xml -c state.xml --group-index {i}\n" for i in range(4)),
        encoding="utf-8")
    done = subprocess.run(
        CLI + ["md-run", "-i", "REST2.in", "-p", "built.pdb", "-groupfile", "group",
               "-ng", "4", "-odir", str(tmp_path / "out")],
        cwd=tmp_path, capture_output=True, text=True, timeout=300, env=environment)
    assert done.returncode != 0, done.stdout
    assert "mpi4py" in done.stderr, done.stderr
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir()), \
        sorted(p.name for p in (tmp_path / "out").iterdir())
